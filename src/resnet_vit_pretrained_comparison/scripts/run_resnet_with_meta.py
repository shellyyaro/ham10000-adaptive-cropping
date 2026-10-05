# src/experiments/run_resnet_with_meta.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import transforms

from resnet_vit_pretrained_comparison.data.AGCWD import build_agcwd_from_cfg
from resnet_vit_pretrained_comparison.data.mask_fusion_loader import FusedMaskLoader
from resnet_vit_pretrained_comparison.models.models_cnn_resnet import HAMMultiInputDataset
from resnet_vit_pretrained_comparison.scripts.common_io import (
    build_run_name_compact,
    get_free_path,
    make_run_dir,
    save_eval_artifacts,
    save_training_plots,
    write_config_json,
    write_best_params_json,
)
from resnet_vit_pretrained_comparison.training.common_imbalance import (
    build_loss_and_logits_transform,
)
from resnet_vit_pretrained_comparison.training.common_training import (
    compute_class_counts,
    compute_imbalanced_metrics,
    forward_flexible,
    set_seed,
    unpack_batch,
)
from resnet_vit_pretrained_comparison.training.training_cnn_resnet import (
    train_single_model,
)


def _debug_history_lengths(history: Any, prefix: str = "[DEBUG]") -> None:
    print(f"\n{prefix} history keys and lengths:")
    if not isinstance(history, dict):
        print(f"{prefix} history is not a dict: {type(history)}")
        return

    for k, v in history.items():
        try:
            print(f"{prefix} {k}: len={len(v)}")
        except Exception:
            print(f"{prefix} {k}: non-sequence ({type(v)})")


def _sanitize_history_for_plots(history: Any) -> Dict[str, List[Any]]:
    if not isinstance(history, dict):
        print(f"[WARN] history is not a dict ({type(history)}). Using empty history for plots.")
        return {}

    lengths = {}
    sanitized = {}

    for k, v in history.items():
        if isinstance(v, (str, bytes)):
            continue

        try:
            lengths[k] = len(v)
        except Exception:
            continue

    if not lengths:
        print("[WARN] history has no sequence-like values. Using empty history for plots.")
        return {}

    min_len = min(lengths.values())
    max_len = max(lengths.values())

    if min_len != max_len:
        print(f"[WARN] inconsistent history lengths detected: min={min_len}, max={max_len}")
        for k, ln in lengths.items():
            if ln != min_len:
                print(f"[WARN] trimming '{k}' from len={ln} to len={min_len}")

    for k, v in history.items():
        if isinstance(v, (str, bytes)):
            sanitized[k] = [v] * min_len
            continue

        try:
            seq = list(v)
            sanitized[k] = seq[:min_len]
        except Exception:
            sanitized[k] = [v] * min_len

    return sanitized


def build_transforms(
    img_size: int = 224,
    rotation_deg: float = 15.0,
    use_color_jitter: bool = True,
    hflip: bool = True,
    pretrained: bool = True,
    use_random_crop: bool = True,
    random_crop_scale: tuple[float, float] = (0.7, 1.0),
    use_gaussian_blur: bool = False,
    gaussian_blur_p: float = 0.3,
    gaussian_blur_sigma: tuple[float, float] = (0.1, 2.0),
    gaussian_blur_kernel: int = 23,
):
    if pretrained:
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
    else:
        norm_mean = [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]

    train_list = []
    if use_random_crop:
        train_list.append(transforms.RandomResizedCrop(img_size, scale=random_crop_scale))
    else:
        train_list.append(transforms.Resize((img_size, img_size)))

    if hflip:
        train_list.append(transforms.RandomHorizontalFlip(p=0.5))
        train_list.append(transforms.RandomVerticalFlip(p=0.2))
    if rotation_deg and rotation_deg > 0:
        train_list.append(transforms.RandomRotation(rotation_deg))
    if use_color_jitter:
        train_list.append(
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02)
        )

    if use_gaussian_blur:
        k = int(gaussian_blur_kernel)
        if k % 2 == 0:
            k += 1
        train_list.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=k, sigma=gaussian_blur_sigma)],
                p=float(gaussian_blur_p),
            )
        )

    train_list += [transforms.ToTensor(), transforms.Normalize(mean=norm_mean, std=norm_std)]

    val_list = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ]
    return transforms.Compose(train_list), transforms.Compose(val_list)


@torch.no_grad()
def evaluate_and_save(
    *,
    model,
    split_df: pd.DataFrame,
    meta_features,
    label2idx: dict,
    batch_size: int,
    val_transform,
    use_masked_image: bool,
    out_dir: Path,
    device: torch.device,
    run_name: str,
    hog_lbp_features=None,
    use_hog_lbp: bool = False,
    no_meta: bool = False,
    logits_tf=None,
    target_class: str = "mel",
    fused_mask_loader: Optional[FusedMaskLoader] = None,
    mask_mode: str = "none",
    agcwd_transform=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval().to(device)

    df_reset = split_df.reset_index(drop=True)

    ds = HAMMultiInputDataset(
        df=df_reset,
        meta_features=meta_features,
        hog_lbp_features=hog_lbp_features,
        transform=val_transform,
        label_col="dx",
        label2idx=label2idx,
        use_masked_image=use_masked_image,
        use_hog_lbp=(use_hog_lbp if not no_meta else False),
        mask_col_name="img_path_masked",
        img_col_name="img_path",
        fused_mask_loader=fused_mask_loader,
        fused_mask_mode=mask_mode,
        rotation_deg=0.0,
        hflip_p=0.0,
        vflip_p=0.0,
        agcwd_transform=agcwd_transform,
    )
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=0)

    use_meta = not bool(no_meta)
    use_mask = (fused_mask_loader is not None and mask_mode == "concat")

    y_true_idx: List[int] = []
    y_pred_idx: List[int] = []

    for batch in loader:
        images, meta_vecs, labels, masks, extras = unpack_batch(batch, device, non_blocking=False)

        outputs = forward_flexible(
            model,
            images,
            meta_vecs,
            use_meta=use_meta,
            masks=masks,
            use_mask=use_mask,
            mask_mode=mask_mode,
        )

        logits = outputs["logits"] if isinstance(outputs, dict) else outputs

        if logits_tf is not None:
            logits = logits_tf(logits)

        preds = logits.argmax(1)
        y_true_idx.extend(labels.cpu().numpy().tolist())
        y_pred_idx.extend(preds.cpu().numpy().tolist())

    paths = save_eval_artifacts(
        model_key="resnet",
        run_name=run_name,
        out_dir=out_dir,
        y_true_idx=y_true_idx,
        y_pred_idx=y_pred_idx,
        label2idx=label2idx,
        split_df=df_reset,
        target_class=target_class,
        title_tag="ResNet-only" if no_meta else "ResNet+Meta",
    )

    acc = float(accuracy_score(y_true_idx, y_pred_idx))
    imb = compute_imbalanced_metrics(
        np.array(y_true_idx), np.array(y_pred_idx), label2idx, target_class=target_class
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


def iter_param_configs(param_grid):
    if isinstance(param_grid, list):
        for cfg in param_grid:
            if not isinstance(cfg, dict):
                raise TypeError(f"param_grid list must contain dicts, got: {type(cfg)}")
            yield cfg
        return

    if not isinstance(param_grid, dict):
        raise TypeError(f"param_grid must be dict or list[dict], got: {type(param_grid)}")

    from itertools import product

    keys = list(param_grid.keys())
    values = []
    for k in keys:
        v = param_grid[k]
        values.append(list(v) if isinstance(v, (list, tuple)) else [v])

    for combo in product(*values):
        yield dict(zip(keys, combo))


def _resolve_consistency_flags(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    שומר על backward compatibility:
    אם לא בוחרים consistency -> שום דבר לא משתנה בבייסליין.
    """
    use_lesion_consistency = bool(cfg.get("use_lesion_consistency", False))
    consistency_weight = float(cfg.get("consistency_weight", 0.0))
    consistency_feature_space = str(cfg.get("consistency_feature_space", "embedding")).lower()
    consistency_loss_type = str(cfg.get("consistency_loss_type", "mse")).lower()
    consistency_pair_sampler = bool(cfg.get("consistency_pair_sampler", False))

    return_features = bool(cfg.get("return_features", False))
    return_lesion_id = bool(cfg.get("return_lesion_id", False))

    if use_lesion_consistency:
        return_features = True
        return_lesion_id = True

    if not use_lesion_consistency:
        consistency_weight = 0.0
        consistency_pair_sampler = False

    if consistency_feature_space not in {"embedding", "logits"}:
        raise ValueError(
            f"Unsupported consistency_feature_space='{consistency_feature_space}'. "
            f"Expected one of: ['embedding', 'logits']"
        )

    if consistency_loss_type not in {"mse", "cosine"}:
        raise ValueError(
            f"Unsupported consistency_loss_type='{consistency_loss_type}'. "
            f"Expected one of: ['mse', 'cosine']"
        )

    return {
        "use_lesion_consistency": use_lesion_consistency,
        "consistency_weight": consistency_weight,
        "consistency_feature_space": consistency_feature_space,
        "consistency_loss_type": consistency_loss_type,
        "consistency_pair_sampler": consistency_pair_sampler,
        "return_features": return_features,
        "return_lesion_id": return_lesion_id,
    }


def run_resnet_experiment(
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
    param_grid: Optional[Dict[str, List[Any]]] = None,
    no_meta: bool = False,
    target_class: str = "mel",
    early_stopping_metric: str = "balanced_acc",
    hog_lbp_features: Optional[np.ndarray] = None,
) -> Path:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    if param_grid is None:
        param_grid = {
            "arch": ["resnet18"],
            "pretrained": [True],
            "img_size": [224],
            "rotation_deg": [15.0],
            "use_color_jitter": [True],
            "hflip": [True],
            "batch_size": [32],
            "lr": [1e-4],
            "weight_decay": [1e-4],
            "num_epochs": [10],
            "patience": [5],
            "img_emb_dim": [512],
            "meta_hidden_dim": [128],
            "use_meta_bn": [True],
            "clf_hidden_dim": [256],
            "dropout": [0.3],
            "stem_kernel_size": [7],
            "stem_stride": [2],
            "stem_use_maxpool": [True],
            "pool_type": ["avg"],
            "gem_p": [3.0],
            "gem_trainable": [False],
            "use_multiscale_stem": [False],
            "stem_branch_kernels": [[3, 3]],
            "stem_branch_dilations": [[1, 2]],
            "use_multiscale_block": [False],
            "multiscale_block_position": ["none"],
            "multiscale_kernels": [[3, 3]],
            "multiscale_dilations": [[1, 2]],

            # NEW: mid-level fusion
            "use_midlevel_fusion": [False],
            "midlevel_source": ["layer2"],
            "midlevel_pool_type": ["avg"],
            "midlevel_emb_dim": [128],

            "use_masked_image": [False],
            "use_hog_lbp": [False],
            "use_fused_masks": [False],
            "mask_mode": ["none"],
            "mask_npz_train": [None],
            "mask_npz_val": [None],
            "mask_npz_test": [None],
            "loss_name": ["ce"],
            "class_weighting": ["none"],
            "label_smoothing": [0.0],
            "logit_adjustment": ["none"],
            "tau": [1.0],
            "sampler": ["none"],
            "focal_gamma": [2.0],
            "focal_alpha_mode": ["none"],
            "agcwd_preset": ["none"],
            "debug_stop_after_model_build": [False],

            # Lesion consistency flags
            "use_lesion_consistency": [False],
            "consistency_weight": [0.0],
            "consistency_feature_space": ["embedding"],
            "consistency_loss_type": ["mse"],
            "consistency_pair_sampler": [False],
            "return_features": [False],
            "return_lesion_id": [False],
        }

    all_labels = (
        set(train_df["dx"].unique())
        | set(val_df["dx"].unique())
        | set(test_df["dx"].unique())
    )
    labels = sorted(all_labels)
    label2idx = {l: i for i, l in enumerate(labels)}
    num_classes = len(labels)

    class_counts_np = compute_class_counts(train_df, label2idx, label_col="dx")

    all_rows = []

    for cfg in iter_param_configs(param_grid):
        consistency_cfg = _resolve_consistency_flags(cfg)
        agcwd_transform = build_agcwd_from_cfg(cfg)

        use_masked = bool(cfg.get("use_masked_image", False))
        if use_masked and ("img_path_masked" not in train_df.columns):
            continue

        cfg_use_hog_lbp = bool(cfg.get("use_hog_lbp", False))
        if cfg_use_hog_lbp and hog_lbp_features is None:
            import warnings

            warnings.warn(
                "Config requests use_hog_lbp=True but hog_lbp_features was not provided. "
                f"Skipping this config: {cfg.get('run_tag', 'unnamed')}",
                UserWarning,
                stacklevel=2,
            )
            continue

        fused_loader_train = None
        fused_loader_val = None
        fused_loader_test = None

        use_fused = bool(cfg.get("use_fused_masks", False))
        mask_mode = str(cfg.get("mask_mode", "none"))

        if use_fused and mask_mode != "none":
            mask_npz_train = cfg.get("mask_npz_train")
            mask_npz_val = cfg.get("mask_npz_val")
            mask_npz_test = cfg.get("mask_npz_test")

            if mask_npz_train and Path(mask_npz_train).exists():
                fused_loader_train = FusedMaskLoader.from_npz(mask_npz_train)
                print(f"✅ Loaded fused masks (train): {fused_loader_train}")

            if mask_npz_val and Path(mask_npz_val).exists():
                fused_loader_val = FusedMaskLoader.from_npz(mask_npz_val)
                print(f"✅ Loaded fused masks (val): {fused_loader_val}")

            if mask_npz_test and Path(mask_npz_test).exists():
                fused_loader_test = FusedMaskLoader.from_npz(mask_npz_test)
                print(f"✅ Loaded fused masks (test): {fused_loader_test}")

        use_spatial_mask = use_fused and mask_mode not in ("none", "concat")

        tf_use_random_crop = False if use_spatial_mask else bool(cfg.get("use_random_crop", True))
        tf_rotation_deg = 0.0 if use_spatial_mask else float(cfg.get("rotation_deg", 15.0))
        tf_hflip = False if use_spatial_mask else bool(cfg.get("hflip", True))

        is_pretrained = bool(cfg.get("pretrained", False))
        train_tf, eval_tf = build_transforms(
            img_size=int(cfg.get("img_size", 224)),
            rotation_deg=tf_rotation_deg,
            use_color_jitter=bool(cfg.get("use_color_jitter", True)),
            hflip=tf_hflip,
            pretrained=is_pretrained,
            use_random_crop=tf_use_random_crop,
            random_crop_scale=tuple(cfg.get("random_crop_scale", (0.7, 1.0))),
            use_gaussian_blur=bool(cfg.get("use_gaussian_blur", False)),
            gaussian_blur_p=float(cfg.get("gaussian_blur_p", 0.3)),
            gaussian_blur_sigma=tuple(cfg.get("gaussian_blur_sigma", (0.1, 2.0))),
            gaussian_blur_kernel=int(cfg.get("gaussian_blur_kernel", 23)),
        )

        run_name = build_run_name_compact(
            model_key="resnet",
            cfg=cfg,
            no_meta=bool(no_meta),
            run_tag=cfg.get("run_tag"),
            arch=cfg.get("arch", "resnet18"),
        )

        run_dir = make_run_dir(out_root, run_name)

        common_extras = {
            "seed": seed,
            "split_seed": split_seed,
            "no_meta": bool(no_meta),
            "target_class": target_class,
            "early_stopping_metric": early_stopping_metric,
            "device": str(device),
        }

        write_config_json(
            run_dir,
            {**cfg, **consistency_cfg},
            extras=common_extras,
            filename="config.json",
        )

        best_params = {
            "run_tag": cfg.get("run_tag", run_name),
            "backbone": "resnet",
            "arch": cfg.get("arch", "resnet18"),
            "pretrained": bool(cfg.get("pretrained", False)),
            "batch_size": int(cfg.get("batch_size", 32)),
            "lr": float(cfg.get("lr", 1e-4)),
            "weight_decay": float(cfg.get("weight_decay", 1e-4)),
            "num_epochs": int(cfg.get("num_epochs", 10)),
            "patience": int(cfg.get("patience", 5)),
            "img_emb_dim": int(cfg.get("img_emb_dim", 512)),
            "meta_hidden_dim": int(cfg.get("meta_hidden_dim", 128)),
            "use_meta_bn": bool(cfg.get("use_meta_bn", True)),
            "clf_hidden_dim": int(cfg.get("clf_hidden_dim", 256)),
            "dropout": float(cfg.get("dropout", 0.3)),
            "use_meta": (not bool(no_meta)),
            "stem_kernel_size": int(cfg.get("stem_kernel_size", 7)),
            "stem_stride": int(cfg.get("stem_stride", 2)),
            "stem_use_maxpool": bool(cfg.get("stem_use_maxpool", True)),
            "pool_type": str(cfg.get("pool_type", "avg")).lower(),
            "gem_p": float(cfg.get("gem_p", 3.0)),
            "gem_trainable": bool(cfg.get("gem_trainable", False)),
            "use_multiscale_stem": bool(cfg.get("use_multiscale_stem", False)),
            "stem_branch_kernels": list(cfg.get("stem_branch_kernels", [3, 3])),
            "stem_branch_dilations": list(cfg.get("stem_branch_dilations", [1, 2])),
            "use_multiscale_block": bool(cfg.get("use_multiscale_block", False)),
            "multiscale_block_position": str(cfg.get("multiscale_block_position", "none")).lower(),
            "multiscale_kernels": list(cfg.get("multiscale_kernels", [3, 3])),
            "multiscale_dilations": list(cfg.get("multiscale_dilations", [1, 2])),

            # mid-level fusion
            "use_midlevel_fusion": bool(cfg.get("use_midlevel_fusion", False)),
            "midlevel_source": str(cfg.get("midlevel_source", "layer2")).lower(),
            "midlevel_pool_type": str(cfg.get("midlevel_pool_type", "avg")).lower(),
            "midlevel_emb_dim": int(cfg.get("midlevel_emb_dim", 128)),

            "loss_name": cfg.get("loss_name", "ce"),
            "class_weighting": cfg.get("class_weighting", "none"),
            "label_smoothing": float(cfg.get("label_smoothing", 0.0)),
            "logit_adjustment": cfg.get("logit_adjustment", "none"),
            "tau": float(cfg.get("tau", 1.0)),
            "sampler": cfg.get("sampler", "none"),
            "focal_gamma": float(cfg.get("focal_gamma", 2.0)),
            "focal_alpha_mode": cfg.get("focal_alpha_mode", "none"),
            "fused_mask_loader": fused_loader_train,
            "mask_mode": mask_mode,
            "rotation_deg": float(cfg.get("rotation_deg", 0.0)) if use_spatial_mask else 0.0,
            "hflip": bool(cfg.get("hflip", False)) if use_spatial_mask else False,
            "vflip": bool(cfg.get("vflip", False)) if use_spatial_mask else False,
            "agcwd_transform": agcwd_transform,
            "debug_stop_after_model_build": bool(cfg.get("debug_stop_after_model_build", False)),

            # Lesion consistency params
            "use_lesion_consistency": consistency_cfg["use_lesion_consistency"],
            "consistency_weight": consistency_cfg["consistency_weight"],
            "consistency_feature_space": consistency_cfg["consistency_feature_space"],
            "consistency_loss_type": consistency_cfg["consistency_loss_type"],
            "consistency_pair_sampler": consistency_cfg["consistency_pair_sampler"],
            "return_features": consistency_cfg["return_features"],
            "return_lesion_id": consistency_cfg["return_lesion_id"],
        }

        write_best_params_json(
            run_dir,
            best_params,
            extras=common_extras,
            filename="best_params_used.json",
        )

        print("\n[RUN DEBUG best_params]")
        for k in [
            "run_tag",
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
            "use_lesion_consistency",
            "consistency_weight",
            "consistency_feature_space",
            "consistency_loss_type",
            "consistency_pair_sampler",
            "return_features",
            "return_lesion_id",
            "debug_stop_after_model_build",
        ]:
            print(f"{k}: cfg={cfg.get(k)} | best_params={best_params.get(k)}")

        model, best_metric_value, history = train_single_model(
            train_df=train_df,
            val_df=val_df,
            meta_features=meta_features,
            meta_feature_dim=meta_feature_dim,
            label2idx=label2idx,
            num_classes=num_classes,
            device=device,
            best_params=best_params,
            train_transform=train_tf,
            val_transform=eval_tf,
            epochs=int(cfg.get("num_epochs", 10)),
            seed=seed,
            hog_lbp_features=hog_lbp_features,
            use_hog_lbp=cfg_use_hog_lbp,
            use_masked_image=use_masked,
            cnn_cfg=None,
            target_class=target_class,
            early_stopping_metric=early_stopping_metric,
            patience=int(cfg.get("patience", 5)),
            class_counts_np=class_counts_np,
        )

        _debug_history_lengths(history, prefix="[DEBUG BEFORE PLOTS]")
        history_for_plots = _sanitize_history_for_plots(history)
        _debug_history_lengths(history_for_plots, prefix="[DEBUG SANITIZED]")

        _, logits_tf_eval, _ = build_loss_and_logits_transform(
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

        tag = "pretrained" if bool(cfg.get("pretrained", False)) else "scratch"
        file_tag = "nometa" if no_meta else "meta"
        mask_tag = f"_{mask_mode}" if (use_fused and mask_mode != "none") else ""
        ckpt = get_free_path(run_dir / f"{cfg.get('arch', 'resnet18')}_{tag}_{file_tag}{mask_tag}.pt")
        torch.save(model.state_dict(), ckpt)

        try:
            plots = save_training_plots(history_for_plots, run_dir, run_name="train")
        except Exception as e:
            print(f"[WARN] save_training_plots failed for run '{run_name}': {e}")
            plots = {}

        eval_val = evaluate_and_save(
            model=model,
            split_df=val_df,
            meta_features=meta_features,
            label2idx=label2idx,
            batch_size=int(cfg.get("batch_size", 32)),
            val_transform=eval_tf,
            use_masked_image=use_masked,
            out_dir=run_dir,
            device=device,
            run_name="val",
            hog_lbp_features=hog_lbp_features,
            use_hog_lbp=cfg_use_hog_lbp,
            no_meta=no_meta,
            logits_tf=logits_tf_eval,
            target_class=target_class,
            fused_mask_loader=fused_loader_val,
            mask_mode=mask_mode,
            agcwd_transform=agcwd_transform,
        )
        eval_test = evaluate_and_save(
            model=model,
            split_df=test_df,
            meta_features=meta_features,
            label2idx=label2idx,
            batch_size=int(cfg.get("batch_size", 32)),
            val_transform=eval_tf,
            use_masked_image=use_masked,
            out_dir=run_dir,
            device=device,
            run_name="test",
            hog_lbp_features=hog_lbp_features,
            use_hog_lbp=cfg_use_hog_lbp,
            no_meta=no_meta,
            logits_tf=logits_tf_eval,
            target_class=target_class,
            fused_mask_loader=fused_loader_test,
            mask_mode=mask_mode,
            agcwd_transform=agcwd_transform,
        )

        all_rows.append(
            {
                "config_name": run_name,
                **cfg,
                **consistency_cfg,
                "seed": seed,
                "split_seed": split_seed,
                "no_meta": bool(no_meta),
                "target_class": target_class,
                "early_stopping_metric": early_stopping_metric,
                "best_metric_value_during_training": float(best_metric_value),
                "weights_path": str(ckpt),
                "config_json_path": str(run_dir / "config.json"),
                "best_params_json_path": str(run_dir / "best_params_used.json"),
                **plots,
                **eval_val,
                **eval_test,
            }
        )

    summary_df = pd.DataFrame(all_rows)
    summary_path = get_free_path(out_root / f"resnet_val_summary_nometa{int(no_meta)}.csv")
    summary_df.to_csv(summary_path, index=False)
    return summary_path
