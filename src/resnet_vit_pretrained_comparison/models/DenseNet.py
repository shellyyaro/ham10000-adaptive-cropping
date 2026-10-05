# resnet_vit_pretrained_comparison/models/DenseNet.py
# DenseNet with Meta features + Fused Mask Support (UPDATED)

from __future__ import annotations

from typing import Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def build_densenet_backbone(
    arch: str = "densenet121",
    pretrained: bool = True,
    in_channels: int = 3,
) -> Tuple[nn.Module, int]:
    """
    Build DenseNet backbone (FEATURES ONLY) with optional 4-channel input.

    Args:
        arch: "densenet121" | "densenet169" | "densenet201"
        pretrained: Whether to use ImageNet pretrained weights
        in_channels: 3 for RGB, 4 for RGB+mask (concat mode)

    Returns:
        features: DenseNet features module (returns (B, C, H, W))
        img_emb_dim: number of output channels C (before pooling)
    """
    if arch == "densenet121":
        if pretrained:
            weights = models.DenseNet121_Weights.IMAGENET1K_V1
            model = models.densenet121(weights=weights)
        else:
            model = models.densenet121(weights=None)
        img_emb_dim = 1024

    elif arch == "densenet169":
        if pretrained:
            weights = models.DenseNet169_Weights.IMAGENET1K_V1
            model = models.densenet169(weights=weights)
        else:
            model = models.densenet169(weights=None)
        img_emb_dim = 1664

    elif arch == "densenet201":
        if pretrained:
            weights = models.DenseNet201_Weights.IMAGENET1K_V1
            model = models.densenet201(weights=weights)
        else:
            model = models.densenet201(weights=None)
        img_emb_dim = 1920

    else:
        raise ValueError(
            f"Unknown arch: {arch}. Options: densenet121, densenet169, densenet201"
        )

    # ✅ Support 4-channel input (RGB + mask) for concat mode
    if in_channels == 4:
        # DenseNet first conv: model.features.conv0 -> Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        original_conv = model.features.conv0
        original_weight = original_conv.weight.data  # (out_ch, 3, k, k)

        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=(original_conv.bias is not None),
        )

        with torch.no_grad():
            # copy RGB weights
            new_conv.weight[:, :3, :, :] = original_weight
            # init mask channel as mean(RGB)
            new_conv.weight[:, 3:4, :, :] = original_weight.mean(dim=1, keepdim=True)

            if original_conv.bias is not None:
                new_conv.bias.data = original_conv.bias.data

        model.features.conv0 = new_conv

        if pretrained:
            warnings.warn(
                f"Created 4-channel {arch} from pretrained 3-channel weights. "
                f"4th channel initialized as mean(RGB). Fine-tuning recommended.",
                UserWarning,
                stacklevel=2,
            )

    elif in_channels != 3:
        raise ValueError(f"in_channels must be 3 or 4, got {in_channels}")

    # ✅ Return FEATURES ONLY (clean backbone)
    features = model.features
    return features, img_emb_dim


class DenseNetWithMeta(nn.Module):
    """
    DenseNet (features) + (optional) meta MLP + classifier.

    ✅ API matches EfficientNetWithMeta:
      - use_meta=True  -> requires meta_vecs
      - use_meta=False -> ignores meta_vecs
      - forward(images, meta_vecs=None)

    ✅ Supports 4-channel input when mask_mode="concat"
    """

    def __init__(
        self,
        num_classes: int,
        arch: str = "densenet121",
        pretrained: bool = True,
        meta_feature_dim: int = 0,
        use_meta: bool = True,
        meta_hidden_dim: int = 128,
        clf_hidden_dim: int = 256,
        dropout: float = 0.3,
        in_channels: int = 3,
    ):
        super().__init__()
        self.use_meta = bool(use_meta)
        self.in_channels = int(in_channels)
        self.arch = str(arch)

        # backbone: returns (B, C, H, W)
        self.backbone, img_emb_dim = build_densenet_backbone(
            arch=self.arch,
            pretrained=bool(pretrained),
            in_channels=self.in_channels,
        )
        self.img_emb_dim = int(img_emb_dim)

        # pool -> (B, C, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # meta net
        if self.use_meta:
            self.meta_net = nn.Sequential(
                nn.Linear(int(meta_feature_dim), int(meta_hidden_dim)),
                nn.BatchNorm1d(int(meta_hidden_dim)),
                nn.ReLU(inplace=True),
            )
            clf_in_dim = self.img_emb_dim + int(meta_hidden_dim)
        else:
            self.meta_net = None
            clf_in_dim = self.img_emb_dim

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(int(clf_in_dim), int(clf_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(clf_hidden_dim), int(num_classes)),
        )

    def forward(self, images: torch.Tensor, meta_vecs: torch.Tensor | None = None) -> torch.Tensor:
        if images.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels}-channel input, got {images.shape[1]}. "
                f"If using mask_mode='concat', create model with in_channels=4."
            )

        # features -> relu -> pool -> flatten
        feat = self.backbone(images)          # (B, C, H, W)
        feat = F.relu(feat, inplace=True)
        feat = self.pool(feat)               # (B, C, 1, 1)
        img_feat = feat.flatten(1)           # (B, C)

        if not self.use_meta:
            return self.classifier(img_feat)

        if meta_vecs is None:
            raise ValueError(
                "use_meta=True but meta_vecs is None. Pass meta_vecs or set use_meta=False."
            )

        meta_feat = self.meta_net(meta_vecs)
        x = torch.cat([img_feat, meta_feat], dim=1)
        return self.classifier(x)


def DenseNetOnly(
    num_classes: int,
    arch: str = "densenet121",
    pretrained: bool = True,
    dropout: float = 0.3,
    in_channels: int = 3,
) -> DenseNetWithMeta:
    """Convenience constructor: DenseNetWithMeta with use_meta=False."""
    return DenseNetWithMeta(
        num_classes=num_classes,
        arch=arch,
        pretrained=pretrained,
        meta_feature_dim=0,
        use_meta=False,
        dropout=dropout,
        in_channels=in_channels,
    )


def densenet_transforms(
    pretrained: bool = True,
    img_size: int = 224,
    rotation_deg: float = 0.0,
    use_color_jitter: bool = False,
    color_jitter_brightness: float = 0.0,
    color_jitter_contrast: float = 0.0,
    color_jitter_saturation: float = 0.0,
    color_jitter_hue: float = 0.0,
    hflip: bool = False,
    vflip: bool = False,
):
    """
    NOTE:
    - OK to use rotation/hflip/vflip ONLY when NOT using masks.
    - When using masks (mask_mode != "none"), do these geometric augs in the Dataset
      (HAMMultiInputDataset) so the mask stays aligned.
    """
    from torchvision import transforms

    if pretrained:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    train_augs = [transforms.Resize((img_size, img_size))]

    if rotation_deg > 0:
        train_augs.append(transforms.RandomRotation(degrees=rotation_deg))

    if use_color_jitter and (
        color_jitter_brightness > 0
        or color_jitter_contrast > 0
        or color_jitter_saturation > 0
        or color_jitter_hue > 0
    ):
        train_augs.append(
            transforms.ColorJitter(
                brightness=color_jitter_brightness,
                contrast=color_jitter_contrast,
                saturation=color_jitter_saturation,
                hue=color_jitter_hue,
            )
        )

    if hflip:
        train_augs.append(transforms.RandomHorizontalFlip(p=0.5))

    if vflip:
        train_augs.append(transforms.RandomVerticalFlip(p=0.5))

    train_augs.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_tf = transforms.Compose(train_augs)

    val_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return train_tf, val_tf