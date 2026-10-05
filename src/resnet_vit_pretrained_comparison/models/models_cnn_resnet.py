from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

from resnet_vit_pretrained_comparison.data.mask_fusion_loader import FusedMaskLoader


# ============================================================
# Tensor-level mask application (after normalization)
# ============================================================

def _apply_mask_to_tensor(
    image: torch.Tensor,
    mask: torch.Tensor,
    mask_mode: str,
    soft_alpha: float = 0.3,
) -> torch.Tensor:
    mode = str(mask_mode).lower()

    if mode in ("multiply", "replace_bg"):
        return image * mask

    elif mode == "soft_multiply":
        return image * (soft_alpha + (1.0 - soft_alpha) * mask)

    elif mode == "blur_bg":
        pad = 11
        blurred = F.avg_pool2d(
            F.pad(image.unsqueeze(0), (pad, pad, pad, pad), mode="reflect"),
            kernel_size=23,
            stride=1,
        ).squeeze(0)
        return mask * image + (1.0 - mask) * blurred

    else:
        raise ValueError(
            f"_apply_mask_to_tensor: unsupported mask_mode='{mask_mode}'. "
            f"Supported: multiply, replace_bg, blur_bg, soft_multiply."
        )


# ============================================================
# Synchronized geometric transform for mask
# ============================================================

def _normalize_mask_np(mask_np: np.ndarray) -> np.ndarray:
    if mask_np is None:
        return None

    mask_np = np.asarray(mask_np)

    if mask_np.size == 0:
        return None

    mask_np = np.squeeze(mask_np)

    if mask_np.ndim == 3:
        if mask_np.shape[0] == 1:
            mask_np = mask_np[0]
        elif mask_np.shape[-1] == 1:
            mask_np = mask_np[..., 0]
        else:
            mask_np = mask_np.mean(axis=-1)

    if mask_np.ndim != 2:
        raise ValueError(f"Mask must become 2D, got shape={mask_np.shape}")

    if mask_np.dtype == np.bool_:
        mask_np = mask_np.astype(np.float32)
    else:
        mask_np = mask_np.astype(np.float32, copy=False)

    mask_np = np.nan_to_num(mask_np, nan=0.0, posinf=1.0, neginf=0.0)

    vmax = float(mask_np.max()) if mask_np.size > 0 else 0.0
    if vmax <= 0.0:
        return np.zeros_like(mask_np, dtype=np.float32)

    if vmax > 1.0:
        mask_np = mask_np / vmax

    return np.clip(mask_np, 0.0, 1.0).astype(np.float32, copy=False)


def _sync_geometric_to_mask(
    mask_np: np.ndarray,
    target_size: tuple,
    hflip: bool,
    vflip: bool,
    angle: float,
) -> torch.Tensor:
    mask_np = _normalize_mask_np(mask_np)

    H_out, W_out = target_size

    if mask_np is None:
        return torch.zeros(1, H_out, W_out, dtype=torch.float32)

    mask_t = torch.from_numpy(mask_np).unsqueeze(0).float()

    if hflip:
        mask_t = torch.flip(mask_t, dims=[-1])
    if vflip:
        mask_t = torch.flip(mask_t, dims=[-2])
    if angle != 0.0:
        mask_t = TF.rotate(mask_t, angle, interpolation=TF.InterpolationMode.BILINEAR)

    if mask_t.shape[-2] != H_out or mask_t.shape[-1] != W_out:
        mask_t = F.interpolate(
            mask_t.unsqueeze(0),
            size=(H_out, W_out),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return mask_t.clamp(0.0, 1.0).to(torch.float32)


# ============================================================
# Dataset
# ============================================================

class HAMMultiInputDataset(Dataset):
    """
    Dataset for HAM10000. Returns (image_tensor, meta_vec, label, mask_tensor).
    """

    VALID_MODES = {"none", "multiply", "concat", "blur_bg", "replace_bg", "soft_multiply"}

    def __init__(
        self,
        df,
        meta_features: np.ndarray,
        hog_lbp_features: Optional[np.ndarray],
        transform,
        label_col: str,
        label2idx: dict,
        use_masked_image: bool = False,
        use_hog_lbp: bool = False,
        mask_col_name: str = "img_path_masked",
        img_col_name: str = "img_path",
        fused_mask_loader: Optional[FusedMaskLoader] = None,
        fused_mask_mode: str = "none",
        rotation_deg: float = 0.0,
        hflip_p: float = 0.5,
        vflip_p: float = 0.0,
        agcwd_transform=None,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.meta_features = meta_features
        self.hog_lbp_features = hog_lbp_features
        self.transform = transform
        self.label_col = label_col
        self.label2idx = label2idx
        self.use_masked_image = use_masked_image
        self.use_hog_lbp = use_hog_lbp and (hog_lbp_features is not None)
        self.mask_col_name = mask_col_name
        self.img_col_name = img_col_name
        self.fused_mask_loader = fused_mask_loader
        self.fused_mask_mode = str(fused_mask_mode).lower()
        self.rotation_deg = float(rotation_deg)
        self.hflip_p = float(hflip_p)
        self.vflip_p = float(vflip_p)
        self.agcwd_transform = agcwd_transform

        if self.fused_mask_mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid fused_mask_mode='{self.fused_mask_mode}'. "
                f"Options: {self.VALID_MODES}"
            )
        if self.img_col_name not in self.df.columns:
            raise ValueError(f"Missing column: {self.img_col_name}")
        if self.label_col not in self.df.columns:
            raise ValueError(f"Missing column: {self.label_col}")
        if "row_idx" not in self.df.columns:
            raise ValueError("Missing column: row_idx")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = str(row[self.img_col_name])

        if (
            self.use_masked_image
            and self.mask_col_name in self.df.columns
            and pd.notna(row[self.mask_col_name])
        ):
            img_path = str(row[self.mask_col_name])

        pil_img = Image.open(img_path).convert("RGB")

        if self.agcwd_transform is not None:
            pil_img = self.agcwd_transform(pil_img)

        active_mode = self.fused_mask_mode
        has_mask = self.fused_mask_loader is not None and active_mode != "none"

        do_hflip = has_mask and (self.hflip_p > 0) and (random.random() < self.hflip_p)
        do_vflip = has_mask and (self.vflip_p > 0) and (random.random() < self.vflip_p)

        angle = 0.0
        if has_mask and self.rotation_deg > 0:
            angle = random.uniform(-self.rotation_deg, self.rotation_deg)

        if has_mask:
            if do_hflip:
                pil_img = TF.hflip(pil_img)
            if do_vflip:
                pil_img = TF.vflip(pil_img)
            if angle != 0.0:
                pil_img = TF.rotate(
                    pil_img,
                    angle,
                    interpolation=TF.InterpolationMode.BILINEAR,
                )

        x = self.transform(pil_img) if self.transform is not None else TF.to_tensor(pil_img)

        fused_mask_tensor: Optional[torch.Tensor] = None

        if self.fused_mask_loader is not None and active_mode != "none":
            image_id = Path(img_path).stem
            mask_np = self.fused_mask_loader.get_mask(image_id)

            if mask_np is not None:
                target_size = (x.shape[1], x.shape[2])

                if active_mode == "concat":
                    fused_mask_tensor = _sync_geometric_to_mask(
                        mask_np=mask_np,
                        target_size=target_size,
                        hflip=do_hflip,
                        vflip=do_vflip,
                        angle=angle,
                    )
                else:
                    mask_t = _sync_geometric_to_mask(
                        mask_np=mask_np,
                        target_size=target_size,
                        hflip=do_hflip,
                        vflip=do_vflip,
                        angle=angle,
                    )
                    x = _apply_mask_to_tensor(x, mask_t, active_mode)
                    fused_mask_tensor = mask_t

        if fused_mask_tensor is None:
            fused_mask_tensor = torch.zeros(1, x.shape[1], x.shape[2], dtype=torch.float32)

        row_idx = int(row["row_idx"])
        meta_vec = self.meta_features[row_idx].astype(np.float32)

        if self.use_hog_lbp:
            hog_vec = self.hog_lbp_features[row_idx].astype(np.float32)
            meta_vec = np.concatenate([meta_vec, hog_vec])

        meta_vec = torch.from_numpy(meta_vec)
        label_idx = torch.tensor(self.label2idx[row[self.label_col]], dtype=torch.long)

        return x, meta_vec, label_idx, fused_mask_tensor


# ============================================================
# ResNet helper modules
# ============================================================

class GeM(nn.Module):
    """Generalized Mean Pooling."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6, trainable: bool = False):
        super().__init__()
        if trainable:
            self.p = nn.Parameter(torch.ones(1) * float(p))
        else:
            self.register_buffer("p", torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x):
        p = torch.clamp(self.p, min=self.eps)
        x = x.clamp(min=self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x.pow(1.0 / p)


class MultiScaleStem(nn.Module):
    """
    Small multi-scale stem:
    branch A: k1 / d1
    branch B: k2 / d2
    concat -> 1x1 fuse -> BN -> ReLU
    Output channels remain the same as the standard stem.
    """

    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 64,
        stride: int = 2,
        kernels=(3, 3),
        dilations=(1, 2),
    ):
        super().__init__()

        k1, k2 = kernels
        d1, d2 = dilations

        if k1 not in (3, 5, 7) or k2 not in (3, 5, 7):
            raise ValueError(f"MultiScaleStem kernels must be in (3,5,7), got {kernels}")

        b1_out = out_ch // 2
        b2_out = out_ch - b1_out

        self.branch1 = nn.Conv2d(
            in_ch,
            b1_out,
            kernel_size=k1,
            stride=stride,
            padding=(k1 // 2) * d1,
            dilation=d1,
            bias=False,
        )
        self.branch2 = nn.Conv2d(
            in_ch,
            b2_out,
            kernel_size=k2,
            stride=stride,
            padding=(k2 // 2) * d2,
            dilation=d2,
            bias=False,
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = torch.cat([x1, x2], dim=1)
        return self.fuse(x)


class MultiScaleFusionBlock(nn.Module):
    """
    Residual multi-scale block with same input/output channels.
    branch A: k1 / d1
    branch B: k2 / d2
    concat -> 1x1 fuse -> residual add
    """

    def __init__(
        self,
        channels: int,
        kernels=(3, 3),
        dilations=(1, 2),
    ):
        super().__init__()

        k1, k2 = kernels
        d1, d2 = dilations

        if k1 not in (3, 5, 7) or k2 not in (3, 5, 7):
            raise ValueError(f"MultiScaleFusionBlock kernels must be in (3,5,7), got {kernels}")

        c1 = channels // 2
        c2 = channels - c1

        self.branch1 = nn.Sequential(
            nn.Conv2d(
                channels,
                c1,
                kernel_size=k1,
                padding=(k1 // 2) * d1,
                dilation=d1,
                bias=False,
            ),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                channels,
                c2,
                kernel_size=k2,
                padding=(k2 // 2) * d2,
                dilation=d2,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        out = self.fuse(out)
        out = out + identity
        return self.relu(out)


# ============================================================
# ResNet Model
# ============================================================

class ResNetWithMeta(nn.Module):
    """ResNet backbone + optional meta features + optional mid-level fusion."""

    def __init__(self, num_classes: int, meta_dim: int = 0, cfg: dict = None):
        super().__init__()
        cfg = cfg or {}

        arch = cfg.get("arch", "resnet18")
        pretrained = cfg.get("pretrained", False)
        img_emb_dim = cfg.get("img_emb_dim", 512)
        meta_hidden_dim = cfg.get("meta_hidden_dim", 128)
        use_meta_bn = cfg.get("use_meta_bn", True)
        clf_hidden_dim = cfg.get("clf_hidden_dim", 256)
        dropout = cfg.get("dropout", 0.3)
        use_meta = cfg.get("use_meta", True)

        stem_kernel_size = int(cfg.get("stem_kernel_size", 7))
        stem_stride = int(cfg.get("stem_stride", 2))
        stem_use_maxpool = bool(cfg.get("stem_use_maxpool", True))

        use_multiscale_stem = bool(cfg.get("use_multiscale_stem", False))
        stem_branch_kernels = tuple(cfg.get("stem_branch_kernels", [3, 3]))
        stem_branch_dilations = tuple(cfg.get("stem_branch_dilations", [1, 2]))

        use_multiscale_block = bool(cfg.get("use_multiscale_block", False))
        multiscale_block_position = str(cfg.get("multiscale_block_position", "none")).lower()
        multiscale_kernels = tuple(cfg.get("multiscale_kernels", [3, 3]))
        multiscale_dilations = tuple(cfg.get("multiscale_dilations", [1, 2]))

        pool_type = str(cfg.get("pool_type", "avg")).lower()
        gem_p = float(cfg.get("gem_p", 3.0))
        gem_trainable = bool(cfg.get("gem_trainable", False))

        # optional mid-level fusion
        use_midlevel_fusion = bool(cfg.get("use_midlevel_fusion", False))
        midlevel_source = str(cfg.get("midlevel_source", "layer2")).lower()
        midlevel_pool_type = str(cfg.get("midlevel_pool_type", "avg")).lower()
        midlevel_emb_dim = int(cfg.get("midlevel_emb_dim", 128))

        if midlevel_source not in {"layer1", "layer2", "layer3"}:
            raise ValueError("midlevel_source must be 'layer1', 'layer2', or 'layer3'")

        self.use_meta = bool(use_meta) and (meta_dim > 0)
        self.use_midlevel_fusion = use_midlevel_fusion
        self.midlevel_source = midlevel_source
        self.midlevel_pool_type = midlevel_pool_type
        self.midlevel_emb_dim = midlevel_emb_dim

        if arch == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet18(weights=weights)
        elif arch == "resnet34":
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet34(weights=weights)
        elif arch == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unknown arch: {arch}")

        if stem_kernel_size not in (3, 5, 7):
            raise ValueError(f"stem_kernel_size must be 3, 5 or 7, got {stem_kernel_size}")

        if stem_stride not in (1, 2):
            raise ValueError(f"stem_stride must be 1 or 2, got {stem_stride}")

        original_conv1 = base.conv1

        if use_multiscale_stem:
            stem_conv = MultiScaleStem(
                in_ch=original_conv1.in_channels,
                out_ch=original_conv1.out_channels,
                stride=stem_stride,
                kernels=stem_branch_kernels,
                dilations=stem_branch_dilations,
            )
            stem_bn = nn.Identity()
            stem_relu = nn.Identity()
        else:
            if stem_kernel_size != 7 or stem_stride != 2:
                new_padding = stem_kernel_size // 2
                new_conv1 = nn.Conv2d(
                    in_channels=original_conv1.in_channels,
                    out_channels=original_conv1.out_channels,
                    kernel_size=stem_kernel_size,
                    stride=stem_stride,
                    padding=new_padding,
                    bias=False,
                )

                with torch.no_grad():
                    if pretrained:
                        if stem_kernel_size == 7:
                            new_conv1.weight.copy_(original_conv1.weight)
                        elif stem_kernel_size == 5:
                            cropped = original_conv1.weight[:, :, 1:6, 1:6]
                            new_conv1.weight.copy_(cropped)
                        elif stem_kernel_size == 3:
                            cropped = original_conv1.weight[:, :, 2:5, 2:5]
                            new_conv1.weight.copy_(cropped)
                    else:
                        nn.init.kaiming_normal_(new_conv1.weight, mode="fan_out", nonlinearity="relu")

                base.conv1 = new_conv1

            stem_conv = base.conv1
            stem_bn = base.bn1
            stem_relu = base.relu

        stem_layers = [stem_conv, stem_bn, stem_relu]

        if stem_use_maxpool:
            stem_layers.append(base.maxpool)
        else:
            stem_layers.append(nn.Identity())

        self.stem = nn.Sequential(*stem_layers)

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.ms_after_layer1 = None
        self.ms_after_layer2 = None

        if use_multiscale_block:
            if multiscale_block_position == "after_layer1":
                layer1_out_channels = self.layer1[-1].conv2.out_channels
                self.ms_after_layer1 = MultiScaleFusionBlock(
                    channels=layer1_out_channels,
                    kernels=multiscale_kernels,
                    dilations=multiscale_dilations,
                )
            elif multiscale_block_position == "after_layer2":
                layer2_out_channels = self.layer2[-1].conv2.out_channels
                self.ms_after_layer2 = MultiScaleFusionBlock(
                    channels=layer2_out_channels,
                    kernels=multiscale_kernels,
                    dilations=multiscale_dilations,
                )
            elif multiscale_block_position == "none":
                pass
            else:
                raise ValueError(
                    "multiscale_block_position must be 'after_layer1', 'after_layer2', or 'none'"
                )

        print(
            "\n[MODEL DEBUG]",
            {
                "run_tag": cfg.get("run_tag"),
                "arch": arch,
                "pretrained": pretrained,
                "use_multiscale_stem": use_multiscale_stem,
                "stem_branch_kernels": stem_branch_kernels,
                "stem_branch_dilations": stem_branch_dilations,
                "use_multiscale_block": use_multiscale_block,
                "multiscale_block_position": multiscale_block_position,
                "multiscale_kernels": multiscale_kernels,
                "multiscale_dilations": multiscale_dilations,
                "stem_kernel_size": stem_kernel_size,
                "stem_stride": stem_stride,
                "stem_use_maxpool": stem_use_maxpool,
                "stem_module_type": type(stem_conv).__name__,
                "stem_bn_type": type(stem_bn).__name__,
                "stem_relu_type": type(stem_relu).__name__,
                "has_ms_after_layer1": self.ms_after_layer1 is not None,
                "has_ms_after_layer2": self.ms_after_layer2 is not None,
                "ms_after_layer1_type": type(self.ms_after_layer1).__name__ if self.ms_after_layer1 is not None else None,
                "ms_after_layer2_type": type(self.ms_after_layer2).__name__ if self.ms_after_layer2 is not None else None,
                "use_midlevel_fusion": self.use_midlevel_fusion,
                "midlevel_source": self.midlevel_source if self.use_midlevel_fusion else None,
                "midlevel_pool_type": self.midlevel_pool_type if self.use_midlevel_fusion else None,
                "midlevel_emb_dim": self.midlevel_emb_dim if self.use_midlevel_fusion else None,
            }
        )

        if pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)
        elif pool_type == "gem":
            self.pool = GeM(p=gem_p, trainable=gem_trainable)
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}. Use: avg / max / gem")

        if self.use_midlevel_fusion:
            if midlevel_pool_type == "avg":
                self.midlevel_pool = nn.AdaptiveAvgPool2d(1)
            elif midlevel_pool_type == "max":
                self.midlevel_pool = nn.AdaptiveMaxPool2d(1)
            elif midlevel_pool_type == "gem":
                self.midlevel_pool = GeM(p=gem_p, trainable=gem_trainable)
            else:
                raise ValueError(
                    f"Unknown midlevel_pool_type: {midlevel_pool_type}. Use: avg / max / gem"
                )
        else:
            self.midlevel_pool = None

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)

            x = self.stem(dummy)

            x = self.layer1(x)
            if self.ms_after_layer1 is not None:
                x = self.ms_after_layer1(x)
            feat_l1 = x

            x = self.layer2(x)
            if self.ms_after_layer2 is not None:
                x = self.ms_after_layer2(x)
            feat_l2 = x

            x = self.layer3(x)
            feat_l3 = x

            x = self.layer4(x)
            deep_feat = x

            deep_feat = self.pool(deep_feat)
            actual_backbone_dim = deep_feat.view(1, -1).shape[1]

            if self.use_midlevel_fusion:
                if self.midlevel_source == "layer1":
                    mid_feat = feat_l1
                elif self.midlevel_source == "layer2":
                    mid_feat = feat_l2
                else:
                    mid_feat = feat_l3

                mid_feat = self.midlevel_pool(mid_feat)
                actual_midlevel_dim = mid_feat.view(1, -1).shape[1]
            else:
                actual_midlevel_dim = 0

        self.img_emb = nn.Sequential(
            nn.Linear(actual_backbone_dim, img_emb_dim),
            nn.ReLU(),
        )

        if self.use_midlevel_fusion:
            self.mid_img_emb = nn.Sequential(
                nn.Linear(actual_midlevel_dim, midlevel_emb_dim),
                nn.ReLU(),
            )
            fused_img_dim = img_emb_dim + midlevel_emb_dim
        else:
            self.mid_img_emb = None
            fused_img_dim = img_emb_dim

        if self.use_meta:
            meta_layers = [nn.Linear(meta_dim, meta_hidden_dim)]
            if use_meta_bn:
                meta_layers.append(nn.BatchNorm1d(meta_hidden_dim))
            meta_layers.append(nn.ReLU(inplace=True))
            self.meta_net = nn.Sequential(*meta_layers)
            clf_in_dim = fused_img_dim + meta_hidden_dim
        else:
            self.meta_net = None
            clf_in_dim = fused_img_dim

        self.classifier = nn.Sequential(
            nn.Linear(clf_in_dim, clf_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(clf_hidden_dim, num_classes),
        )

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(
            "[MODEL PARAMS]",
            {
                "run_tag": cfg.get("run_tag"),
                "actual_backbone_dim": actual_backbone_dim,
                "actual_midlevel_dim": actual_midlevel_dim if self.use_midlevel_fusion else None,
                "img_emb_dim": img_emb_dim,
                "midlevel_emb_dim": midlevel_emb_dim if self.use_midlevel_fusion else None,
                "fused_img_dim": fused_img_dim,
                "clf_hidden_dim": clf_hidden_dim,
                "meta_dim": meta_dim,
                "use_meta": self.use_meta,
                "total_params": total_params,
                "trainable_params": trainable_params,
            }
        )

    def _extract_backbone_maps(self, images):
        x = self.stem(images)

        x = self.layer1(x)
        if self.ms_after_layer1 is not None:
            x = self.ms_after_layer1(x)
        feat_l1 = x

        x = self.layer2(x)
        if self.ms_after_layer2 is not None:
            x = self.ms_after_layer2(x)
        feat_l2 = x

        x = self.layer3(x)
        feat_l3 = x

        x = self.layer4(x)
        deep_feat = x

        return {
            "layer1": feat_l1,
            "layer2": feat_l2,
            "layer3": feat_l3,
            "deep": deep_feat,
        }

    def forward_features(self, images):
        feats = self._extract_backbone_maps(images)

        deep_feat = self.pool(feats["deep"])
        deep_feat = deep_feat.view(deep_feat.size(0), -1)

        if not self.use_midlevel_fusion:
            return deep_feat

        deep_emb = self.img_emb(deep_feat)

        mid_feat = feats[self.midlevel_source]
        mid_feat = self.midlevel_pool(mid_feat)
        mid_feat = mid_feat.view(mid_feat.size(0), -1)
        mid_emb = self.mid_img_emb(mid_feat)

        return torch.cat([deep_emb, mid_emb], dim=1)

    def forward(self, images, meta_vecs=None):
        feat = self.forward_features(images)

        if self.use_midlevel_fusion:
            img_emb = feat
        else:
            img_emb = self.img_emb(feat)

        if not self.use_meta:
            return self.classifier(img_emb)

        if meta_vecs is None:
            raise ValueError("use_meta=True but meta_vecs is None")

        meta_emb = self.meta_net(meta_vecs)
        return self.classifier(torch.cat([img_emb, meta_emb], dim=1))


# ============================================================
# FlexibleCNN
# ============================================================

class FlexibleCNNWithMeta(nn.Module):
    """Simple CNN with optional meta features (for baseline experiments)."""

    def __init__(self, num_classes: int, meta_dim: int = 0, cfg: dict = None):
        super().__init__()
        cfg = cfg or {}

        img_emb_dim = cfg.get("img_emb_dim", 256)
        num_blocks = cfg.get("num_blocks", 4)
        base_channels = cfg.get("base_channels", 32)
        use_maxpool = cfg.get("use_maxpool", True)
        use_batchnorm = cfg.get("use_batchnorm", True)
        meta_hidden_dim = cfg.get("meta_hidden_dim", 128)
        use_meta_bn = cfg.get("use_meta_bn", True)
        clf_hidden_dim = cfg.get("clf_hidden_dim", 256)
        dropout = cfg.get("dropout", 0.3)
        use_meta = cfg.get("use_meta", True)

        self.use_meta = bool(use_meta) and (meta_dim > 0)

        layers = []
        in_ch = 3
        out_ch = base_channels
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            if use_maxpool:
                layers.append(nn.MaxPool2d(2))
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512)

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.img_emb = nn.Sequential(nn.Linear(in_ch, img_emb_dim), nn.ReLU())

        if self.use_meta:
            meta_layers = [nn.Linear(meta_dim, meta_hidden_dim)]
            if use_meta_bn:
                meta_layers.append(nn.BatchNorm1d(meta_hidden_dim))
            meta_layers.append(nn.ReLU(inplace=True))
            self.meta_net = nn.Sequential(*meta_layers)
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

    def forward(self, images, meta_vecs=None):
        feat = self.features(images)
        feat = self.pool(feat)
        feat = feat.view(feat.size(0), -1)
        img_emb = self.img_emb(feat)

        if not self.use_meta:
            return self.classifier(img_emb)

        if meta_vecs is None:
            raise ValueError("use_meta=True but meta_vecs is None")

        meta_emb = self.meta_net(meta_vecs)
        return self.classifier(torch.cat([img_emb, meta_emb], dim=1))


# ============================================================
# ResNetWithMetaLegacy — for loading old checkpoints
# ============================================================

class ResNetWithMetaLegacy(nn.Module):
    """
    Legacy ResNet architecture (pre-img_emb).
    Needed to load checkpoints saved before img_emb was added.
    """

    def __init__(self, num_classes: int, meta_dim: int = 0, cfg: dict = None):
        super().__init__()
        cfg = cfg or {}

        arch = cfg.get("arch", "resnet18")
        pretrained = cfg.get("pretrained", False)
        meta_hidden_dim = cfg.get("meta_hidden_dim", 128)
        use_meta_bn = cfg.get("use_meta_bn", True)
        clf_hidden_dim = cfg.get("clf_hidden_dim", 256)
        dropout = cfg.get("dropout", 0.3)
        use_meta = cfg.get("use_meta", True)

        self.use_meta = bool(use_meta) and (meta_dim > 0)

        if arch == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet18(weights=weights)
        elif arch == "resnet34":
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet34(weights=weights)
        elif arch == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unknown arch: {arch}")

        self.img_backbone = nn.Sequential(*list(base.children())[:-1])

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            backbone_out = self.img_backbone(dummy)
            backbone_dim = backbone_out.view(1, -1).shape[1]

        if self.use_meta:
            meta_layers = [nn.Linear(meta_dim, meta_hidden_dim)]
            if use_meta_bn:
                meta_layers.append(nn.BatchNorm1d(meta_hidden_dim))
            meta_layers.append(nn.ReLU(inplace=True))
            self.meta_net = nn.Sequential(*meta_layers)
            clf_in_dim = backbone_dim + meta_hidden_dim
        else:
            self.meta_net = None
            clf_in_dim = backbone_dim

        self.classifier = nn.Sequential(
            nn.Linear(clf_in_dim, clf_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(clf_hidden_dim, num_classes),
        )

    def forward(self, images, meta_vecs=None):
        feat = self.img_backbone(images)
        feat = feat.view(feat.size(0), -1)

        if not self.use_meta:
            return self.classifier(feat)

        if meta_vecs is None:
            raise ValueError("use_meta=True but meta_vecs is None")

        meta_emb = self.meta_net(meta_vecs)
        return self.classifier(torch.cat([feat, meta_emb], dim=1))