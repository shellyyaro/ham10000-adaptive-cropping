from __future__ import annotations

import warnings
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt backbone that returns a flat image embedding vector.

    Supports:
      - convnext_tiny
      - convnext_small
      - convnext_base

    Also supports 4-channel input for mask concat mode.
    """

    def __init__(
        self,
        arch: str = "convnext_tiny",
        pretrained: bool = True,
        in_channels: int = 3,
    ):
        super().__init__()

        if arch == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.convnext_tiny(weights=weights)
            img_emb_dim = 768

        elif arch == "convnext_small":
            weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.convnext_small(weights=weights)
            img_emb_dim = 768

        elif arch == "convnext_base":
            weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.convnext_base(weights=weights)
            img_emb_dim = 1024

        else:
            raise ValueError(
                f"Unknown arch: {arch}. "
                f"Options: convnext_tiny, convnext_small, convnext_base"
            )

        # Support 4-channel input for concat mask mode
        if in_channels == 4:
            # first block is Conv2dNormActivation, first item is Conv2d
            original_conv = model.features[0][0]
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
                new_conv.weight[:, :3, :, :] = original_weight
                new_conv.weight[:, 3:4, :, :] = original_weight.mean(dim=1, keepdim=True)
                if original_conv.bias is not None:
                    new_conv.bias.data = original_conv.bias.data

            model.features[0][0] = new_conv

            if pretrained:
                warnings.warn(
                    f"Created 4-channel {arch} from pretrained 3-channel weights. "
                    f"4th channel initialized as mean(RGB). Fine-tuning recommended.",
                    UserWarning,
                    stacklevel=2,
                )

        elif in_channels != 3:
            raise ValueError(f"in_channels must be 3 or 4, got {in_channels}")

        self.features = model.features
        self.avgpool = model.avgpool

        # classifier in torchvision ConvNeXt is:
        # Sequential(LayerNorm2d(...), Flatten(1), Linear(...))
        # We keep the normalization part and do flatten manually.
        if isinstance(model.classifier, nn.Sequential) and len(model.classifier) >= 1:
            self.norm = model.classifier[0]
        else:
            self.norm = nn.Identity()

        self.img_emb_dim = img_emb_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)          # (B, C, H, W)
        x = self.avgpool(x)           # (B, C, 1, 1)
        x = self.norm(x)              # (B, C, 1, 1)
        x = torch.flatten(x, 1)       # (B, C)
        return x


class ConvNeXtWithMeta(nn.Module):
    """
    ConvNeXt backbone + optional meta MLP + classifier.

    API matches EfficientNetWithMeta / ResNetWithMeta:
      - forward(images, meta_vecs=None)
      - use_meta=True  -> expects meta_vecs
      - use_meta=False -> ignores meta_vecs
    """

    def __init__(
        self,
        num_classes: int,
        arch: str = "convnext_tiny",
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
        self.arch = arch

        self.backbone = ConvNeXtBackbone(
            arch=arch,
            pretrained=pretrained,
            in_channels=in_channels,
        )
        img_emb_dim = self.backbone.img_emb_dim
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
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(clf_hidden_dim, num_classes),
        )

    def forward(self, images: torch.Tensor, meta_vecs: torch.Tensor | None = None) -> torch.Tensor:
        if images.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels}-channel input, got {images.shape[1]}. "
                f"If using mask_mode='concat', create model with in_channels=4."
            )

        img_feat = self.backbone(images)

        if not self.use_meta:
            return self.classifier(img_feat)

        if meta_vecs is None:
            raise ValueError(
                "use_meta=True but meta_vecs is None. "
                "Pass meta_vecs or set use_meta=False."
            )

        meta_feat = self.meta_net(meta_vecs)
        x = torch.cat([img_feat, meta_feat], dim=1)
        return self.classifier(x)


def ConvNeXtOnly(
    num_classes: int,
    arch: str = "convnext_tiny",
    pretrained: bool = True,
    dropout: float = 0.3,
    in_channels: int = 3,
) -> ConvNeXtWithMeta:
    return ConvNeXtWithMeta(
        num_classes=num_classes,
        arch=arch,
        pretrained=pretrained,
        meta_feature_dim=0,
        use_meta=False,
        dropout=dropout,
        in_channels=in_channels,
    )


def convnext_transforms(
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
    from torchvision import transforms

    if pretrained:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    train_augs = [
        transforms.Resize((img_size, img_size)),
    ]

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

    train_augs.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_tf = transforms.Compose(train_augs)

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_tf, val_tf