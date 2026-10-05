# src/experiments/run_vit_with_meta.py
# ✅ UPDATED (v3.3): AGCWD preprocessing support added and applied to train/val/test.
#   AGCWDTransform is applied at dataset level (in __getitem__, right after Image.open),
#   before any geometric or color augmentation. Same instance used for train/val/test.
#
# v3.2 recap:
#   NoMeta is DataLoader-safe (meta_dummy tensor), dynamic model_key in eval save.
#   Geometric augmentations synced between image and mask.
#   Mask applied at tensor level after normalization (consistent with ResNet).
#   using_mask based on actual loader (not just use_fused flag).

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import accuracy_score
from torch import amp
from torch.utils.data import DataLoader, Dataset

from resnet_vit_pretrained_comparison.data.AGCWD import build_agcwd_from_cfg
from resnet_vit_pretrained_comparison.data.mask_fusion_loader import FusedMaskLoader
from resnet_vit_pretrained_comparison.models.models_cnn_resnet import (
    HAMMultiInputDataset,
    _apply_mask_to_tensor,
    _sync_geometric_to_mask,
)
from resnet_vit_pretrained_comparison.models.models_vit import (
    build_vit_backbone,
    vit_transforms,
)
from resnet_vit_pretrained_comparison.scripts.common_io import (
    build_run_name_compact,
    get_free_path,
    make_run_dir,
    save_eval_artifacts,
    save_training_plots,
    write_config_json,
)
from resnet_vit_pretrained_comparison.training.common_imbalance import (
    build_loss_and_logits_transform,
)
from resnet_vit_pretrained_comparison.training.common_training import (
    compute_class_counts,
    compute_imbalanced_metrics,
    forward_flexible,
    get_val_metric_from_history,
    make_weighted_sampler,
    metric_direction,
    set_seed,
    unpack_batch,
)


# -------------------------
# Datasets
# -------------------------
class SimpleImageOnlyDataset(Dataset):
    """
    Dataset for image-only (no metadata) experiments.

    For spatial mask modes (multiply, blur_bg, soft_multiply, replace_bg),
    the mask is applied to the NORMALIZED IMAGE TENSOR after transform
    (via _apply_mask_to_tensor), NOT to the PIL image. This is consistent
    with HAMMultiInputDataset and ResNet mask handling.

    Geometric augmentations (flip, rotate) are applied to the PIL image
    manually BEFORE self.transform, and the same ops are applied to the
    mask via _sync_geometric_to_mask — keeping image and mask aligned.

    self.transform must NOT include flip/rotation/RRCrop when mask_mode
    != "none" (use use_spatial_mask=True in vit_transforms). Geometric ops
    are handled exclusively here with synchronized mask transforms.

    concat mode: mask returned as (1,H,W) tensor after sync, not applied
    to image. forward_flexible concatenates it as the 4th channel.

    agcwd_transform: optional AGCWDTransform instance applied immediately
    after Image.open(), before any geometric or color augmentation.
    """

    VALID_MODES = {"none", "multiply", "concat", "blur_bg", "replace_bg", "soft_multiply"}

    def __init__(
        self,
        df: pd.DataFrame,
        transform,
        label_col: str,
        label2idx: dict,
        img_col_name: str = "img_path",
        fused_mask_loader: Optional[FusedMaskLoader] = None,
        fused_mask_mode: str = "none",
        rotation_deg: float = 0.0,
        hflip_p: float = 0.0,
        vflip_p: float = 0.0,
        agcwd_transform=None,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform
        self.label_col = label_col
        self.label2idx = label2idx
        self.img_col_name = img_col_name
        self.fused_mask_loader = fused_mask_loader
        self.fused_mask_mode = str(fused_mask_mode).lower()
        self.rotation_deg = float(rotation_deg)
        self.hflip_p = float(hflip_p)
        self.vflip_p = float(vflip_p)
        self.agcwd_transform = agcwd_transform

        if self.fused_mask_mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid fused_mask_mode='{self.fused_mask_mode}'. Options: {self.VALID_MODES}"
            )

        if self.img_col_name not in self.df.columns:
            raise ValueError(f"Expected column '{self.img_col_name}' in df")
        if self.label_col not in self.df.columns:
            raise ValueError(f"Expected column '{self.label_col}' in df")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = str(row[self.img_col_name])
        label_str = row[self.label_col]
        y = int(self.label2idx[label_str])

        pil_img = Image.open(img_path).convert("RGB")

        # AGCWD before any geometric or color transform
        if self.agcwd_transform is not None:
            pil_img = self.agcwd_transform(pil_img)

        active_mode = self.fused_mask_mode

        # has_mask is True for concat too (needs synced geometry)
        has_mask = (self.fused_mask_loader is not None) and (active_mode != "none")

        # Sample geometric params for synchronized transform
        do_hflip = has_mask and (self.hflip_p > 0) and (random.random() < self.hflip_p)
        do_vflip = has_mask and (self.vflip_p > 0) and (random.random() < self.vflip_p)
        angle = 0.0
        if has_mask and self.rotation_deg > 0:
            angle = random.uniform(-self.rotation_deg, self.rotation_deg)

        # Apply geometric ops to PIL before self.transform
        if has_mask:
            if do_hflip:
                pil_img = TF.hflip(pil_img)
            if do_vflip:
                pil_img = TF.vflip(pil_img)
            if angle != 0.0:
                pil_img = TF.rotate(pil_img, angle, interpolation=TF.InterpolationMode.BILINEAR)

        x = self.transform(pil_img) if self.transform is not None else TF.to_tensor(pil_img)

        fused_mask_tensor: Optional[torch.Tensor] = None

        if self.fused_mask_loader is not None and active_mode != "none":
            image_id = Path(img_path).stem
            mask_np = self.fused_mask_loader.get_mask(image_id)
            if mask_np is not None:
                target_size = (x.shape[1], x.shape[2])

                mask_t = _sync_geometric_to_mask(
                    mask_np=mask_np,
                    target_size=target_size,
                    hflip=do_hflip,
                    vflip=do_vflip,
                    angle=angle,
                )

                if active_mode == "concat":
                    fused_mask_tensor = mask_t
                else:
                    x = _apply_mask_to_tensor(x, mask_t, active_mode)
                    fused_mask_tensor = mask_t

        if fused_mask_tensor is None:
            fused_mask_tensor = torch.zeros(1, x.shape[1], x.shape[2], dtype=torch.float32)

        label_tensor = torch.tensor(y, dtype=torch.long)

        # Never return None in a batch tuple
        meta_dummy = torch.empty(0, dtype=torch.float32)

        return x, meta_dummy, label_tensor, fused_mask_tensor


# -------------------------
# Models
# -------------------------
class ViTWithMeta(nn.Module):
    """
    ViT backbone + (optional) meta MLP + classifier.
    Supports 4-channel input when mask_mode="concat".
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool,
        meta_feature_dim: int,
        use_meta: bool = True,
        meta_hidden_dim: int = 128,
        clf_hidden_dim: int = 256,
        dropout: float = 0.3,
        in_channels: int = 3,
    ):
        super().__init__()
        self.use_meta = bool(use_meta)
        self.in_channels = int(in_channels)

        self.vit, img_emb_dim = build_vit_backbone(pretrained=pretrained, in_channels=self.in_channels)
        self.img_emb_dim = img_emb_dim

        if self.use_meta:
            self.meta_net = nn.Sequential(
                nn.Linear(meta_feature_dim, meta_hidden_dim),
                nn.BatchNorm1d(meta_hidden_dim),
                nn.ReLU(inplace=True),
            )
            clf_in_dim = img_emb_dim + meta_hidden_dim
        else:
            self.meta_net = None
            clf_in_dim = img_emb_dim

        self.classifier = nn.Sequential(
            nn.Linear(clf_in_dim, clf_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(clf_hidden_dim, num_classes),
        )

    def forward(self, images: torch.Tensor, meta_vecs: torch.Tensor | None = None) -> torch.Tensor:
        if images.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels}-channel input, got {images.shape[1]}. "
                f"If using mask_mode='concat', create model with in_channels=4."
            )

        img_feat = self.vit(images)

        if not self.use_meta:
            return self.classifier(img_feat)

        if meta_vecs is None:
            raise ValueError("use_meta=True but meta_vecs is None.")

        meta_feat = self.meta_net(meta_vecs)
        x = torch.cat([img_feat, meta_feat], dim=1)
        return self.classifier(x)


def ViTOnly(num_classes: int, pretrained: bool, dropout: float = 0.3, in_channels: int = 3) -> ViTWithMeta:
    return ViTWithMeta(
        num_classes=num_classes,
        pretrained=pretrained,
        meta_feature_dim=0,
        use_meta=False,
        dropout=dropout,
        in_channels=in_channels,
    )


# -------------------------
# Training
# -------------------------
def train_vit_with_meta(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    *,
    train_logits_tf: Callable,
    eval_logits_tf: Callable,
    loss_fn: nn.Module,
    label2idx: dict,
    target_class: str = "mel",
    early_stopping_metric: str = "balanced_acc",
    use_amp: bool = True,
    seed: int = 0,
    use_meta: bool = True,
    use_mask: bool = False,
    mask_mode: str = "multiply",
):
    set_seed(seed)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "train_balanced_acc": [],
        "train_macro_f1": [],
        f"train_{target_class}_recall": [],
        f"train_{target_class}_precision": [],
        "val_loss": [],
        "val_acc": [],
        "val_balanced_acc": [],
        "val_macro_f1": [],
        f"val_{target_class}_recall": [],
        f"val_{target_class}_precision": [],
    }

    direction = metric_direction(early_stopping_metric)
    best_metric_value = np.inf if direction == "min" else -np.inf
    best_state = None
    epochs_no_improve = 0

    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss_sum, tr_n = 0.0, 0
        tr_true, tr_pred = [], []

        for batch in train_loader:
            images, meta_vecs, labels, masks = unpack_batch(batch, device)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type=autocast_device, enabled=(use_amp and device.type == "cuda")):
                raw_logits = forward_flexible(
                    model,
                    images,
                    meta_vecs,
                    use_meta=use_meta,
                    masks=masks,
                    use_mask=use_mask,
                    mask_mode=mask_mode,
                )
                logits = train_logits_tf(raw_logits)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_loss_sum += float(loss.item()) * labels.size(0)
            tr_n += labels.size(0)
            tr_true.append(labels.detach().cpu().numpy())
            tr_pred.append(raw_logits.argmax(1).detach().cpu().numpy())

        tr_true = np.concatenate(tr_true)
        tr_pred = np.concatenate(tr_pred)
        train_loss = tr_loss_sum / tr_n if tr_n > 0 else 0.0
        train_acc = float(accuracy_score(tr_true, tr_pred))
        train_imb = compute_imbalanced_metrics(tr_true, tr_pred, label2idx, target_class)

        model.eval()
        va_n = 0
        va_raw_loss_sum = 0.0
        va_true, va_pred = [], []

        with torch.no_grad():
            for batch in val_loader:
                images, meta_vecs, labels, masks = unpack_batch(batch, device)
                with amp.autocast(device_type=autocast_device, enabled=(use_amp and device.type == "cuda")):
                    base_logits = forward_flexible(
                        model,
                        images,
                        meta_vecs,
                        use_meta=use_meta,
                        masks=masks,
                        use_mask=use_mask,
                        mask_mode=mask_mode,
                    )

                    logits_eval = eval_logits_tf(base_logits)
                    preds_eval = logits_eval.argmax(1)

                    raw_logits = train_logits_tf(base_logits)
                    raw_loss = loss_fn(raw_logits, labels)

                va_raw_loss_sum += float(raw_loss.item()) * labels.size(0)
                va_n += labels.size(0)
                va_true.append(labels.detach().cpu().numpy())
                va_pred.append(preds_eval.detach().cpu().numpy())

        va_true = np.concatenate(va_true)
        va_pred = np.concatenate(va_pred)
        val_loss_raw = va_raw_loss_sum / va_n if va_n > 0 else 0.0
        val_acc = float(accuracy_score(va_true, va_pred))
        val_imb = compute_imbalanced_metrics(va_true, va_pred, label2idx, target_class)

        history["epoch"].append(ep)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss_raw)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_balanced_acc"].append(train_imb["balanced_acc"])
        history["val_balanced_acc"].append(val_imb["balanced_acc"])
        history["train_macro_f1"].append(train_imb["macro_f1"])
        history["val_macro_f1"].append(val_imb["macro_f1"])
        history[f"train_{target_class}_recall"].append(train_imb[f"{target_class}_recall"])
        history[f"val_{target_class}_recall"].append(val_imb[f"{target_class}_recall"])
        history[f"train_{target_class}_precision"].append(train_imb[f"{target_class}_precision"])
        history[f"val_{target_class}_precision"].append(val_imb[f"{target_class}_precision"])

        print(
            f"[seed {seed}] epoch {ep}/{epochs} "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, train_bal_acc={train_imb['balanced_acc']:.3f}, "
            f"train_f1={train_imb['macro_f1']:.3f} | "
            f"val_loss={val_loss_raw:.4f}, val_acc={val_acc:.3f}, val_bal_acc={val_imb['balanced_acc']:.3f}, "
            f"val_f1={val_imb['macro_f1']:.3f}"
        )

        current_metric_value = get_val_metric_from_history(
            history,
            early_stopping_metric,
            target_class=target_class,
        )

        improved = (
            (current_metric_value < best_metric_value)
            if direction == "min"
            else (current_metric_value > best_metric_value)
        )

        if improved:
            best_metric_value = float(current_metric_value)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= int(patience):
                print(f"[seed {seed}] Early stopping after {ep} epochs (no improvement for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"[seed {seed}] best {early_stopping_metric} = {best_metric_value:.4f}")
    return model, float(best_metric_value), history


# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def evaluate_and_save_common(
    *,
    model: nn.Module,
    split_df: pd.DataFrame,
    meta_features: np.ndarray,
    label2idx: dict,
    batch_size: int,
    val_transform,
    out_dir: Path,
    run_name: str,
    device: torch.device,
    logits_tf,
    no_meta: bool = False,
    target_class: str = "mel",
    fused_mask_loader: Optional[FusedMaskLoader] = None,
    mask_mode: str = "none",
    model_key: str = "vit",
    agcwd_transform=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval().to(device)
    df_reset = split_df.reset_index(drop=True)

    use_mask = (fused_mask_loader is not None) and (mask_mode == "concat")

    if no_meta:
        ds = SimpleImageOnlyDataset(
            df=df_reset,
            transform=val_transform,
            label_col="dx",
            label2idx=label2idx,
            img_col_name="img_path",
            fused_mask_loader=fused_mask_loader,
            fused_mask_mode=mask_mode,
            rotation_deg=0.0,
            hflip_p=0.0,
            vflip_p=0.0,
            agcwd_transform=agcwd_transform,
        )
    else:
        ds = HAMMultiInputDataset(
            df=df_reset,
            meta_features=meta_features,
            hog_lbp_features=None,
            transform=val_transform,
            label_col="dx",
            label2idx=label2idx,
            use_masked_image=False,
            use_hog_lbp=False,
            mask_col_name="img_path_masked",
            img_col_name="img_path",
            fused_mask_loader=fused_mask_loader,
            fused_mask_mode=mask_mode,
            rotation_deg=0.0,
            hflip_p=0.0,
            vflip_p=0.0,
            agcwd_transform=agcwd_transform,
        )

    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    use_meta = not bool(no_meta)

    y_true_idx: List[int] = []
    y_pred_idx: List[int] = []

    for batch in loader:
        images, meta_vecs, labels, masks = unpack_batch(batch, device)
        logits = forward_flexible(
            model,
            images,
            meta_vecs,
            use_meta=use_meta,
            masks=masks,
            use_mask=use_mask,
            mask_mode=mask_mode,
        )
        logits = logits_tf(logits)
        preds = logits.argmax(1)

        y_true_idx.extend(labels.detach().cpu().numpy().tolist())
        y_pred_idx.extend(preds.detach().cpu().numpy().tolist())

    paths = save_eval_artifacts(
        model_key=str(model_key),
        run_name=run_name,
        out_dir=out_dir,
        y_true_idx=y_true_idx,
        y_pred_idx=y_pred_idx,
        label2idx=label2idx,
        split_df=df_reset,
        target_class=target_class,
    )

    acc = float(accuracy_score(y_true_idx, y_pred_idx))
    imb = compute_imbalanced_metrics(
        np.array(y_true_idx),
        np.array(y_pred_idx),
        label2idx,
        target_class=target_class,
    )

    return {
        f"{run_name}_accuracy": acc,
        f"{run_name}_balanced_accuracy": float(imb["balanced_acc"]),
        f"{run_name}_macro_f1": float(imb["macro_f1"]),
        f"{run_name}_{target_class}_recall": float(imb[f"{target_class}_recall"]),
        f"{run_name}_{target_class}_precision": float(imb[f"{target_class}_precision"]),
        f"{run_name}_n_samples": int(len(y_true_idx)),
        f"{run_name}_preds_path": paths["preds_path"],
        f"{run_name}_classification_report_path": paths["classification_report_path"],
        f"{run_name}_cm_csv_path": paths["cm_csv_path"],
        f"{run_name}_cm_png_path": paths["cm_png_path"],
    }


# -------------------------
# Experiment runner
# -------------------------
def run_vit_experiment(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    meta_features: np.ndarray,
    meta_feature_dim: int,
    out_root: Path,
    device: torch.device,
    seed: int = 0,
    split_seed: int = 0,
    param_grid: Optional[List[Dict[str, Any]]] = None,
    no_meta: bool = False,
    target_class: str = "mel",
    early_stopping_metric: str = "balanced_acc",
) -> Path:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    all_labels = sorted(
        set(train_df["dx"].unique())
        | set(val_df["dx"].unique())
        | set(test_df["dx"].unique())
    )
    label2idx = {l: i for i, l in enumerate(all_labels)}
    num_classes = len(label2idx)

    if param_grid is None:
        param_grid = [
            {
                "run_tag": "vit_default",
                "pretrained": True,
                "img_size": 224,
                "batch_size": 32,
                "lr": 5e-5,
                "weight_decay": 0.05,
                "num_epochs": 10,
                "patience": 3,
                "meta_hidden_dim": 128,
                "clf_hidden_dim": 256,
                "dropout": 0.3,
                "use_fused_masks": False,
                "mask_mode": "none",
                "mask_npz_train": None,
                "mask_npz_val": None,
                "mask_npz_test": None,
                "loss_name": "ce",
                "class_weighting": "none",
                "label_smoothing": 0.0,
                "logit_adjustment": "none",
                "tau": 1.0,
                "sampler": "none",
                "focal_gamma": 2.0,
                "focal_alpha_mode": "none",
                "rotation_deg": 0.0,
                "hflip": False,
                "vflip": False,
                "use_color_jitter": False,
                "agcwd_preset": "none",
            }
        ]

    class_counts_np = compute_class_counts(train_df, label2idx, label_col="dx")
    all_rows = []

    for cfg in param_grid:
        agcwd_transform = build_agcwd_from_cfg(cfg)

        fused_loader_train = None
        fused_loader_val = None
        fused_loader_test = None

        use_fused = bool(cfg.get("use_fused_masks", False))
        mask_mode = str(cfg.get("mask_mode", "none")).lower()

        if use_fused and mask_mode != "none":
            mask_npz_train = cfg.get("mask_npz_train")
            mask_npz_val = cfg.get("mask_npz_val")
            mask_npz_test = cfg.get("mask_npz_test")

            if mask_npz_train and Path(mask_npz_train).exists():
                fused_loader_train = FusedMaskLoader.from_npz(mask_npz_train)
                print(f"Loaded fused masks (train): {fused_loader_train}")
            else:
                print(f"Warning: mask_npz_train not found: {mask_npz_train}")

            if mask_npz_val and Path(mask_npz_val).exists():
                fused_loader_val = FusedMaskLoader.from_npz(mask_npz_val)
                print(f"Loaded fused masks (val): {fused_loader_val}")
            else:
                print(f"Warning: mask_npz_val not found: {mask_npz_val}")

            if mask_npz_test and Path(mask_npz_test).exists():
                fused_loader_test = FusedMaskLoader.from_npz(mask_npz_test)
                print(f"Loaded fused masks (test): {fused_loader_test}")
            else:
                print(f"Warning: mask_npz_test not found: {mask_npz_test}")

        using_mask = (fused_loader_train is not None) and (mask_mode != "none")

        train_tf, eval_tf = vit_transforms(
            pretrained=bool(cfg["pretrained"]),
            img_size=int(cfg["img_size"]),
            rotation_deg=0.0 if using_mask else float(cfg.get("rotation_deg", 0.0)),
            hflip=False if using_mask else bool(cfg.get("hflip", False)),
            vflip=False if using_mask else bool(cfg.get("vflip", False)),
            use_color_jitter=bool(cfg.get("use_color_jitter", False)),
            color_jitter_brightness=float(cfg.get("color_jitter_brightness", 0.1)),
            color_jitter_contrast=float(cfg.get("color_jitter_contrast", 0.1)),
            color_jitter_saturation=float(cfg.get("color_jitter_saturation", 0.05)),
            color_jitter_hue=float(cfg.get("color_jitter_hue", 0.02)),
            use_spatial_mask=using_mask,
        )

        train_logits_tf, eval_logits_tf, loss_fn = build_loss_and_logits_transform(
            loss_name=str(cfg.get("loss_name", "ce")),
            class_weighting=str(cfg.get("class_weighting", "none")),
            label_smoothing=float(cfg.get("label_smoothing", 0.0)),
            focal_gamma=float(cfg.get("focal_gamma", 2.0)),
            focal_alpha_mode=str(cfg.get("focal_alpha_mode", "none")),
            logit_adjustment=str(cfg.get("logit_adjustment", "none")),
            tau=float(cfg.get("tau", 1.0)),
            class_counts_np=class_counts_np,
            device=device,
        )

        run_name = build_run_name_compact(
            model_key="vit",
            cfg=cfg,
            no_meta=bool(no_meta),
            run_tag=cfg.get("run_tag"),
        )
        run_dir = make_run_dir(out_root, run_name)

        write_config_json(
            run_dir,
            cfg,
            extras={
                "seed": seed,
                "split_seed": split_seed,
                "no_meta": bool(no_meta),
                "target_class": target_class,
                "early_stopping_metric": early_stopping_metric,
                "model_type": "vit",
                "device": device,
            },
            filename="config.json",
        )

        ds_hflip_p = 0.5 if (using_mask and bool(cfg.get("hflip", False))) else 0.0
        ds_vflip_p = 0.2 if (using_mask and bool(cfg.get("vflip", False))) else 0.0
        ds_rot_deg = float(cfg.get("rotation_deg", 0.0)) if using_mask else 0.0

        if no_meta:
            ds_train = SimpleImageOnlyDataset(
                df=train_df,
                transform=train_tf,
                label_col="dx",
                label2idx=label2idx,
                img_col_name="img_path",
                fused_mask_loader=fused_loader_train,
                fused_mask_mode=mask_mode,
                rotation_deg=ds_rot_deg,
                hflip_p=ds_hflip_p,
                vflip_p=ds_vflip_p,
                agcwd_transform=agcwd_transform,
            )
            ds_val = SimpleImageOnlyDataset(
                df=val_df,
                transform=eval_tf,
                label_col="dx",
                label2idx=label2idx,
                img_col_name="img_path",
                fused_mask_loader=fused_loader_val,
                fused_mask_mode=mask_mode,
                rotation_deg=0.0,
                hflip_p=0.0,
                vflip_p=0.0,
                agcwd_transform=agcwd_transform,
            )
        else:
            ds_train = HAMMultiInputDataset(
                df=train_df,
                meta_features=meta_features,
                hog_lbp_features=None,
                transform=train_tf,
                label_col="dx",
                label2idx=label2idx,
                use_masked_image=False,
                use_hog_lbp=False,
                mask_col_name="img_path_masked",
                img_col_name="img_path",
                fused_mask_loader=fused_loader_train,
                fused_mask_mode=mask_mode,
                rotation_deg=ds_rot_deg,
                hflip_p=ds_hflip_p,
                vflip_p=ds_vflip_p,
                agcwd_transform=agcwd_transform,
            )
            ds_val = HAMMultiInputDataset(
                df=val_df,
                meta_features=meta_features,
                hog_lbp_features=None,
                transform=eval_tf,
                label_col="dx",
                label2idx=label2idx,
                use_masked_image=False,
                use_hog_lbp=False,
                mask_col_name="img_path_masked",
                img_col_name="img_path",
                fused_mask_loader=fused_loader_val,
                fused_mask_mode=mask_mode,
                rotation_deg=0.0,
                hflip_p=0.0,
                vflip_p=0.0,
                agcwd_transform=agcwd_transform,
            )

        sampler_mode = str(cfg.get("sampler", "none")).lower()
        sampler = None
        shuffle = True
        if sampler_mode == "weighted":
            sampler = make_weighted_sampler(train_df, label2idx, label_col="dx")
            shuffle = False

        g = torch.Generator()
        g.manual_seed(seed)

        train_loader = DataLoader(
            ds_train,
            batch_size=int(cfg["batch_size"]),
            shuffle=shuffle,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
            generator=g,
            drop_last=True,
        )
        val_loader = DataLoader(
            ds_val,
            batch_size=int(cfg["batch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        use_meta = not bool(no_meta)
        in_channels = 4 if (mask_mode == "concat") else 3

        model = ViTWithMeta(
            num_classes=num_classes,
            pretrained=bool(cfg["pretrained"]),
            meta_feature_dim=int(meta_feature_dim) if use_meta else 0,
            use_meta=use_meta,
            meta_hidden_dim=int(cfg.get("meta_hidden_dim", 128)),
            clf_hidden_dim=int(cfg.get("clf_hidden_dim", 256)),
            dropout=float(cfg["dropout"]),
            in_channels=in_channels,
        )

        use_mask = (fused_loader_train is not None and mask_mode == "concat")

        model, best_metric, history = train_vit_with_meta(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=int(cfg["num_epochs"]),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
            patience=int(cfg["patience"]),
            train_logits_tf=train_logits_tf,
            eval_logits_tf=eval_logits_tf,
            loss_fn=loss_fn,
            label2idx=label2idx,
            target_class=target_class,
            early_stopping_metric=early_stopping_metric,
            use_amp=True,
            seed=seed,
            use_meta=use_meta,
            use_mask=use_mask,
            mask_mode=mask_mode,
        )

        tag = "pretrained" if bool(cfg["pretrained"]) else "scratch"
        meta_file_tag = "nometa" if no_meta else "meta"
        mask_tag = f"_{mask_mode}" if (use_fused and mask_mode != "none") else ""
        ckpt = get_free_path(run_dir / f"vit_b16_{tag}_{meta_file_tag}{mask_tag}.pt")

        def safe_torch_save(state_dict, path: Path, retries: int = 3, sleep_sec: float = 0.5):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            last_err = None
            for _ in range(retries):
                try:
                    torch.save(state_dict, tmp, _use_new_zipfile_serialization=False)
                    os.replace(tmp, path)
                    return
                except Exception as e:
                    last_err = e
                    try:
                        if tmp.exists():
                            tmp.unlink()
                    except Exception:
                        pass
                    time.sleep(sleep_sec)
            raise last_err

        safe_torch_save(model.state_dict(), ckpt)

        plots = save_training_plots(history, run_dir, run_name="train")

        eval_val = evaluate_and_save_common(
            model=model,
            split_df=val_df,
            meta_features=meta_features,
            label2idx=label2idx,
            batch_size=int(cfg["batch_size"]),
            val_transform=eval_tf,
            out_dir=run_dir,
            run_name="val",
            device=device,
            logits_tf=eval_logits_tf,
            no_meta=no_meta,
            target_class=target_class,
            fused_mask_loader=fused_loader_val,
            mask_mode=mask_mode,
            model_key="vit",
            agcwd_transform=agcwd_transform,
        )
        eval_test = evaluate_and_save_common(
            model=model,
            split_df=test_df,
            meta_features=meta_features,
            label2idx=label2idx,
            batch_size=int(cfg["batch_size"]),
            val_transform=eval_tf,
            out_dir=run_dir,
            run_name="test",
            device=device,
            logits_tf=eval_logits_tf,
            no_meta=no_meta,
            target_class=target_class,
            fused_mask_loader=fused_loader_test,
            mask_mode=mask_mode,
            model_key="vit",
            agcwd_transform=agcwd_transform,
        )

        all_rows.append(
            {
                "config_name": run_name,
                **cfg,
                "seed": seed,
                "split_seed": split_seed,
                "no_meta": bool(no_meta),
                "target_class": target_class,
                "best_metric_value": float(best_metric),
                "early_stopping_metric": early_stopping_metric,
                "weights_path": str(ckpt),
                **plots,
                **eval_val,
                **eval_test,
            }
        )

    summary_df = pd.DataFrame(all_rows)
    summary_path = get_free_path(out_root / f"vit_val_summary_nometa{int(no_meta)}.csv")
    summary_df.to_csv(summary_path, index=False)
    return summary_path
