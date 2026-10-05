from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import amp
from torch.utils.data import DataLoader

from resnet_vit_pretrained_comparison.data.mask_fusion_loader import FusedMaskLoader
from resnet_vit_pretrained_comparison.models.models_cnn_resnet import (
    FlexibleCNNWithMeta,
    HAMMultiInputDataset,
    ResNetWithMeta,
)
from resnet_vit_pretrained_comparison.training.common_imbalance import (
    build_loss_and_logits_transform,
)
from resnet_vit_pretrained_comparison.training.common_training import (
    compute_class_counts,
    compute_imbalanced_metrics,
    forward_flexible,
    make_weighted_sampler,
    set_seed,
    unpack_batch,
)


# -------------------------
# Helpers
# -------------------------
def compute_per_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label2idx: dict[str, int],
) -> dict[str, float]:
    """
    Return per-label precision / recall / f1 for all labels in label2idx.
    Output keys:
        <label>_precision
        <label>_recall
        <label>_f1
    """
    labels_sorted = sorted(label2idx.items(), key=lambda x: x[1])
    class_indices = [idx for _, idx in labels_sorted]
    class_names = [name for name, _ in labels_sorted]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_indices,
        average=None,
        zero_division=0,
    )

    out = {}
    for i, label_name in enumerate(class_names):
        out[f"{label_name}_precision"] = float(precision[i])
        out[f"{label_name}_recall"] = float(recall[i])
        out[f"{label_name}_f1"] = float(f1[i])

    return out


# -------------------------
# Train / Eval loops
# -------------------------
def train_one_epoch(
    model,
    dataloader,
    loss_fn,
    optimizer,
    device,
    logits_tf,
    label2idx,
    target_class="mel",
    *,
    scaler,
    use_meta: bool = True,
    use_mask: bool = False,
    mask_mode: str = "multiply",
):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    use_amp = scaler.is_enabled() and device.type == "cuda"

    for batch in dataloader:
        images, meta_vecs, labels, masks, extras = unpack_batch(
            batch,
            device,
            non_blocking=False,
        )

        optimizer.zero_grad(set_to_none=True)

        with amp.autocast(device_type=device.type, enabled=use_amp):
            raw_logits = forward_flexible(
                model,
                images,
                meta_vecs,
                use_meta=use_meta,
                masks=masks,
                use_mask=use_mask,
                mask_mode=mask_mode,
            )
            logits = logits_tf(raw_logits)
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item()) * labels.size(0)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    total = len(all_labels)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = float((all_preds == all_labels).sum() / total) if total > 0 else 0.0
    imb_metrics = compute_imbalanced_metrics(all_labels, all_preds, label2idx, target_class)
    per_label_metrics = compute_per_label_metrics(all_labels, all_preds, label2idx)

    return epoch_loss, epoch_acc, imb_metrics, per_label_metrics


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    loss_fn,
    device,
    logits_tf,
    label2idx,
    target_class="mel",
    *,
    use_meta: bool = True,
    raw_logits_tf=None,
    use_mask: bool = False,
    mask_mode: str = "multiply",
):
    model.eval()
    running_loss = 0.0
    running_raw_loss = 0.0
    all_preds = []
    all_labels = []

    use_amp = device.type == "cuda"

    for batch in dataloader:
        images, meta_vecs, labels, masks, extras = unpack_batch(
            batch,
            device,
            non_blocking=False,
        )

        with amp.autocast(device_type=device.type, enabled=use_amp):
            base_logits = forward_flexible(
                model,
                images,
                meta_vecs,
                use_meta=use_meta,
                masks=masks,
                use_mask=use_mask,
                mask_mode=mask_mode,
            )

            logits = logits_tf(base_logits)
            loss = loss_fn(logits, labels)

            if raw_logits_tf is not None:
                raw_logits = raw_logits_tf(base_logits)
                raw_loss = loss_fn(raw_logits, labels)

        running_loss += float(loss.item()) * labels.size(0)
        if raw_logits_tf is not None:
            running_raw_loss += float(raw_loss.item()) * labels.size(0)

        preds = logits.argmax(dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    total = len(all_labels)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = float((all_preds == all_labels).sum() / total) if total > 0 else 0.0
    raw_epoch_loss = (
        running_raw_loss / total
        if raw_logits_tf is not None and total > 0
        else epoch_loss
    )
    imb_metrics = compute_imbalanced_metrics(all_labels, all_preds, label2idx, target_class)
    per_label_metrics = compute_per_label_metrics(all_labels, all_preds, label2idx)

    return epoch_loss, epoch_acc, imb_metrics, per_label_metrics, raw_epoch_loss


def train_single_model(
    train_df,
    val_df,
    meta_features,
    meta_feature_dim,
    label2idx,
    num_classes,
    device,
    best_params,
    train_transform,
    val_transform,
    epochs: int = 5,
    seed: int = 0,
    hog_lbp_features=None,
    use_hog_lbp: bool = False,
    use_masked_image: bool = False,
    cnn_cfg: dict | None = None,
    target_class: str = "mel",
    early_stopping_metric: str = "balanced_acc",
    patience: int = 5,
    class_counts_np: np.ndarray | None = None,
):
    set_seed(seed)

    batch_size = int(best_params.get("batch_size", 32))
    num_epochs = int(best_params.get("num_epochs", epochs))

    loss_name = best_params.get("loss_name", "ce")
    class_weighting = best_params.get("class_weighting", "none")
    label_smoothing = float(best_params.get("label_smoothing", 0.0))
    logit_adjustment = best_params.get("logit_adjustment", "none")
    tau = float(best_params.get("tau", 1.0))
    focal_gamma = float(best_params.get("focal_gamma", 2.0))
    focal_alpha_mode = best_params.get("focal_alpha_mode", "none")
    sampler_mode = best_params.get("sampler", "none")

    fused_mask_loader: Optional[FusedMaskLoader] = best_params.get("fused_mask_loader")
    mask_mode = str(best_params.get("mask_mode", "none"))
    agcwd_transform = best_params.get("agcwd_transform", None)

    debug_stop_after_model_build = bool(best_params.get("debug_stop_after_model_build", False))

    print("\n[TRAIN DEBUG best_params]")
    for k in [
        "run_tag",
        "backbone",
        "arch",
        "pretrained",
        "use_meta",
        "use_multiscale_stem",
        "stem_branch_kernels",
        "stem_branch_dilations",
        "use_multiscale_block",
        "multiscale_block_position",
        "multiscale_kernels",
        "multiscale_dilations",
        "use_midlevel_fusion",
        "midlevel_source",
        "midlevel_pool_type",
        "midlevel_emb_dim",
        "debug_stop_after_model_build",
    ]:
        print(f"{k}: {best_params.get(k)}")

    if best_params.get("use_meta", True):
        hog_dim = 0
        if use_hog_lbp and hog_lbp_features is not None:
            hog_dim = int(hog_lbp_features.shape[1])
        meta_dim_for_model = int(meta_feature_dim) + hog_dim
    else:
        meta_dim_for_model = 0
        use_hog_lbp = False

    backbone_type = best_params.get("backbone", "flexible_cnn")

    if cnn_cfg is None:
        if backbone_type == "resnet":
            cnn_cfg = {
                "run_tag": best_params.get("run_tag"),
                "arch": best_params.get("arch", "resnet18"),
                "pretrained": bool(best_params.get("pretrained", False)),
                "img_emb_dim": int(best_params.get("img_emb_dim", 512)),
                "meta_hidden_dim": int(best_params.get("meta_hidden_dim", 128)),
                "use_meta_bn": bool(best_params.get("use_meta_bn", True)),
                "clf_hidden_dim": int(best_params.get("clf_hidden_dim", 256)),
                "dropout": float(best_params.get("dropout", 0.3)),
                "use_meta": bool(best_params.get("use_meta", True)),

                "stem_kernel_size": int(best_params.get("stem_kernel_size", 7)),
                "stem_stride": int(best_params.get("stem_stride", 2)),
                "stem_use_maxpool": bool(best_params.get("stem_use_maxpool", True)),
                "pool_type": str(best_params.get("pool_type", "avg")).lower(),
                "gem_p": float(best_params.get("gem_p", 3.0)),
                "gem_trainable": bool(best_params.get("gem_trainable", False)),

                "use_multiscale_stem": bool(best_params.get("use_multiscale_stem", False)),
                "stem_branch_kernels": list(best_params.get("stem_branch_kernels", [3, 3])),
                "stem_branch_dilations": list(best_params.get("stem_branch_dilations", [1, 2])),
                "use_multiscale_block": bool(best_params.get("use_multiscale_block", False)),
                "multiscale_block_position": str(
                    best_params.get("multiscale_block_position", "none")
                ).lower(),
                "multiscale_kernels": list(best_params.get("multiscale_kernels", [3, 3])),
                "multiscale_dilations": list(best_params.get("multiscale_dilations", [1, 2])),

                # NEW: mid-level fusion / multi-scale feature fusion
                "use_midlevel_fusion": bool(best_params.get("use_midlevel_fusion", False)),
                "midlevel_source": str(best_params.get("midlevel_source", "layer2")).lower(),
                "midlevel_pool_type": str(best_params.get("midlevel_pool_type", "avg")).lower(),
                "midlevel_emb_dim": int(best_params.get("midlevel_emb_dim", 128)),
            }
        else:
            cnn_cfg = {
                "img_emb_dim": int(best_params.get("img_emb_dim", 256)),
                "num_blocks": int(best_params.get("num_blocks", 4)),
                "base_channels": int(best_params.get("base_channels", 32)),
                "use_maxpool": bool(best_params.get("use_maxpool", True)),
                "use_batchnorm": bool(best_params.get("use_batchnorm", True)),
                "meta_hidden_dim": int(best_params.get("meta_hidden_dim", 128)),
                "use_meta_bn": bool(best_params.get("use_meta_bn", True)),
                "clf_hidden_dim": int(best_params.get("clf_hidden_dim", 256)),
                "dropout": float(best_params.get("dropout", 0.3)),
                "use_meta": bool(best_params.get("use_meta", True)),
            }

    print("\n[TRAIN DEBUG cnn_cfg passed to model]")
    for k, v in cnn_cfg.items():
        print(f"{k}: {v}")

    use_spatial_mask = (
        fused_mask_loader is not None
        and mask_mode not in ("none", "concat")
    )
    train_rotation_deg = float(best_params.get("rotation_deg", 0.0)) if use_spatial_mask else 0.0
    train_hflip_p = 0.5 if (use_spatial_mask and best_params.get("hflip", False)) else 0.0
    train_vflip_p = 0.2 if (use_spatial_mask and best_params.get("vflip", False)) else 0.0

    train_dataset = HAMMultiInputDataset(
        df=train_df,
        meta_features=meta_features,
        hog_lbp_features=hog_lbp_features,
        transform=train_transform,
        label_col="dx",
        label2idx=label2idx,
        use_masked_image=use_masked_image,
        use_hog_lbp=use_hog_lbp,
        mask_col_name="img_path_masked",
        img_col_name="img_path",
        fused_mask_loader=fused_mask_loader,
        fused_mask_mode=mask_mode,
        rotation_deg=train_rotation_deg,
        hflip_p=train_hflip_p,
        vflip_p=train_vflip_p,
        agcwd_transform=agcwd_transform,
    )
    val_dataset = HAMMultiInputDataset(
        df=val_df,
        meta_features=meta_features,
        hog_lbp_features=hog_lbp_features,
        transform=val_transform,
        label_col="dx",
        label2idx=label2idx,
        use_masked_image=use_masked_image,
        use_hog_lbp=use_hog_lbp,
        mask_col_name="img_path_masked",
        img_col_name="img_path",
        fused_mask_loader=fused_mask_loader,
        fused_mask_mode=mask_mode,
        rotation_deg=0.0,
        hflip_p=0.0,
        vflip_p=0.0,
        agcwd_transform=agcwd_transform,
    )

    sampler = None
    shuffle = True
    if str(sampler_mode).lower() == "weighted":
        sampler = make_weighted_sampler(train_df, label2idx, label_col="dx")
        shuffle = False

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    if backbone_type == "resnet":
        model = ResNetWithMeta(
            num_classes=num_classes,
            meta_dim=meta_dim_for_model,
            cfg=cnn_cfg,
        ).to(device)
    else:
        model = FlexibleCNNWithMeta(
            num_classes=num_classes,
            meta_dim=meta_dim_for_model,
            cfg=cnn_cfg,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n[TRAIN DEBUG model built]")
    print("model_type:", type(model).__name__)
    print("total_params:", total_params)
    print("trainable_params:", trainable_params)

    if debug_stop_after_model_build:
        raise SystemExit(
            f"Stopped after model build for debug. run_tag={best_params.get('run_tag')}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(best_params.get("lr", 1e-4)),
        weight_decay=float(best_params.get("weight_decay", 1e-4)),
    )

    if class_counts_np is None:
        class_counts_np = compute_class_counts(train_df, label2idx, label_col="dx")

    train_logits_tf, eval_logits_tf, loss_fn = build_loss_and_logits_transform(
        loss_name=str(loss_name),
        class_weighting=str(class_weighting),
        label_smoothing=float(label_smoothing),
        focal_gamma=float(focal_gamma),
        focal_alpha_mode=str(focal_alpha_mode),
        logit_adjustment=str(logit_adjustment),
        tau=float(tau),
        class_counts_np=class_counts_np,
        device=device,
    )

    scaler = amp.GradScaler(enabled=(device.type == "cuda"))

    use_meta = bool(best_params.get("use_meta", True))
    use_mask = (fused_mask_loader is not None and mask_mode == "concat")

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

    for label_name in sorted(label2idx.keys(), key=lambda x: label2idx[x]):
        history[f"train_{label_name}_precision"] = []
        history[f"train_{label_name}_recall"] = []
        history[f"train_{label_name}_f1"] = []
        history[f"val_{label_name}_precision"] = []
        history[f"val_{label_name}_recall"] = []
        history[f"val_{label_name}_f1"] = []

    best_metric_value = -np.inf if early_stopping_metric != "loss" else np.inf
    best_state_dict = None
    epochs_no_improve = 0

    skip_per_label_dup_keys = {
        f"{target_class}_recall",
        f"{target_class}_precision",
    }

    for epoch in range(num_epochs):
        train_loss, train_acc, train_imb, train_per_label = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            train_logits_tf,
            label2idx,
            target_class,
            scaler=scaler,
            use_meta=use_meta,
            use_mask=use_mask,
            mask_mode=mask_mode,
        )
        val_loss, val_acc, val_imb, val_per_label, val_loss_raw = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            eval_logits_tf,
            label2idx,
            target_class,
            use_meta=use_meta,
            raw_logits_tf=train_logits_tf,
            use_mask=use_mask,
            mask_mode=mask_mode,
        )

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["train_balanced_acc"].append(float(train_imb["balanced_acc"]))
        history["train_macro_f1"].append(float(train_imb["macro_f1"]))
        history[f"train_{target_class}_recall"].append(float(train_imb[f"{target_class}_recall"]))
        history[f"train_{target_class}_precision"].append(float(train_imb[f"{target_class}_precision"]))

        history["val_loss"].append(float(val_loss_raw))
        history["val_acc"].append(float(val_acc))
        history["val_balanced_acc"].append(float(val_imb["balanced_acc"]))
        history["val_macro_f1"].append(float(val_imb["macro_f1"]))
        history[f"val_{target_class}_recall"].append(float(val_imb[f"{target_class}_recall"]))
        history[f"val_{target_class}_precision"].append(float(val_imb[f"{target_class}_precision"]))

        for key, value in train_per_label.items():
            if key in skip_per_label_dup_keys:
                continue
            history[f"train_{key}"].append(float(value))

        for key, value in val_per_label.items():
            if key in skip_per_label_dup_keys:
                continue
            history[f"val_{key}"].append(float(value))

        print(
            f"[seed {seed}] epoch {epoch + 1}/{num_epochs} "
            f"loss={train_loss:.4f}, acc={train_acc:.3f}, bal_acc={train_imb['balanced_acc']:.3f}, "
            f"f1={train_imb['macro_f1']:.3f} | "
            f"val_loss={val_loss_raw:.4f}, val_acc={val_acc:.3f}, val_bal_acc={val_imb['balanced_acc']:.3f}, "
            f"val_f1={val_imb['macro_f1']:.3f}"
        )

        metric_key_map = {
            "loss": "val_loss",
            "acc": "val_acc",
            "balanced_acc": "val_balanced_acc",
            "macro_f1": "val_macro_f1",
            f"{target_class}_recall": f"val_{target_class}_recall",
            f"{target_class}_precision": f"val_{target_class}_precision",
        }

        key = metric_key_map.get(early_stopping_metric)
        if key is None:
            raise ValueError(
                f"Unknown early_stopping_metric={early_stopping_metric}. Options: {list(metric_key_map)}"
            )

        current_metric_value = history[key][-1]

        improved = (
            current_metric_value < best_metric_value
            if early_stopping_metric == "loss"
            else current_metric_value > best_metric_value
        )

        if improved:
            best_metric_value = current_metric_value
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"[seed {seed}] Early stopping after {epoch + 1} epochs "
                    f"(no improvement for {patience} epochs)"
                )
                break

    print(f"[seed {seed}] best {early_stopping_metric} = {best_metric_value:.3f}")
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, best_metric_value, history