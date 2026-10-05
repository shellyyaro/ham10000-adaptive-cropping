# src/models/models_vit.py
# ✅ עדכון: תמיכה מלאה במסכות פיוז'ן (4-channel input support)
#
# FIX (v2): vit_transforms now accepts use_spatial_mask parameter.
# When use_spatial_mask=True (any mask_mode != "none"):
#   - RandomResizedCrop is DISABLED (replaced with plain Resize).
#     RRCrop zooms into a random sub-region; the mask is not cropped
#     accordingly -> spatial misalignment. For spatial modes (multiply etc.)
#     the mask is already burned into the PIL before transform, so RRCrop
#     is safe there, but for concat it breaks alignment completely.
#     Disabling RRCrop for all masked modes is the safest choice.
#   - hflip/vflip/rotation are forced to False/0 in the transform.
#     Geometric ops are handled exclusively by the Dataset with synchronized
#     mask transforms (_sync_geometric_to_mask). Doing them in both places
#     would apply them twice (and unsynchronized for concat).

import warnings
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import transforms
from typing import Tuple


def build_vit(num_classes: int, pretrained: bool = True, dropout: float = 0.1) -> nn.Module:
    """
    Build a ViT-B/16 with a classification head (Dropout + Linear).
    Use this when you want a standalone ViT classifier.

    .. deprecated::
        Use ``build_vit_backbone`` + ``ViTWithMeta(use_meta=False)`` instead.
    """
    warnings.warn(
        "build_vit is deprecated. Use build_vit_backbone + ViTWithMeta(use_meta=False) instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if pretrained:
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
    else:
        model = vit_b_16(weights=None)

    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_vit_backbone(pretrained: bool = True, in_channels: int = 3) -> Tuple[nn.Module, int]:
    """
    Build a ViT-B/16 *backbone* -- classification head replaced by Identity.

    ✅ NEW: Supports 4-channel input for mask concatenation mode.

    Args:
        pretrained: Whether to use ImageNet pretrained weights
        in_channels: Number of input channels (3 for RGB, 4 for RGB+mask)

    Returns:
        model       : nn.Module  -- ViT whose forward returns [B, img_emb_dim]
        img_emb_dim : int        -- size of the embedding vector (768 for ViT-B/16)

    Usage (in ViTWithMeta):
        vit, img_emb_dim = build_vit_backbone(pretrained=True, in_channels=3)
    """
    if pretrained:
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
    else:
        model = vit_b_16(weights=None)

    # Extract embedding dim from the ORIGINAL head (always a plain nn.Linear)
    img_emb_dim = model.heads.head.in_features

    # ✅ NEW: Support 4-channel input (RGB + mask)
    if in_channels == 4:
        # Get original conv1 weights (3, patch_size, patch_size)
        original_conv = model.conv_proj
        original_weight = original_conv.weight.data  # (768, 3, 16, 16)

        # Create new conv layer with 4 input channels
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=(original_conv.bias is not None)
        )

        # Initialize: copy RGB weights, init mask channel with mean of RGB
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = original_weight
            new_conv.weight[:, 3:4, :, :] = original_weight.mean(dim=1, keepdim=True)

            if original_conv.bias is not None:
                new_conv.bias.data = original_conv.bias.data

        model.conv_proj = new_conv

        if pretrained:
            warnings.warn(
                f"Created 4-channel ViT from pretrained 3-channel weights. "
                f"4th channel initialized as mean(RGB). Fine-tuning recommended.",
                UserWarning,
                stacklevel=2
            )

    elif in_channels != 3:
        raise ValueError(f"in_channels must be 3 or 4, got {in_channels}")

    # Replace head with Identity so forward() returns the embedding
    model.heads.head = nn.Identity()

    return model, img_emb_dim


def vit_transforms(
    pretrained: bool = True,
    img_size: int = 224,
    rotation_deg: float = 0.0,
    hflip: bool = False,
    vflip: bool = False,
    use_color_jitter: bool = False,
    color_jitter_brightness: float = 0.1,
    color_jitter_contrast: float = 0.1,
    color_jitter_saturation: float = 0.05,
    color_jitter_hue: float = 0.02,
    use_spatial_mask: bool = False,
):
    """
    Build train/val transforms for ViT.

    Args:
        use_spatial_mask: Set True when mask_mode != "none".
            When True:
              - RandomResizedCrop is replaced with plain Resize.
                RRCrop is incompatible with mask alignment for concat mode,
                and unnecessarily duplicates the crop for spatial modes
                where the mask is already burned into the PIL.
              - hflip/vflip/rotation are disabled in the transform.
                Geometric augmentations are handled exclusively by the
                Dataset (SimpleImageOnlyDataset / HAMMultiInputDataset)
                via synchronized _sync_geometric_to_mask. Applying them
                here too would double-apply them (and unsynchronized for
                concat, breaking alignment entirely).
            When False (no mask): original behavior — RRCrop + geometric
            augmentations live in the transform as before.
    """
    if pretrained:
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        mean = weights.transforms().mean
        std = weights.transforms().std
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    aug_layers = []

    if use_spatial_mask:
        # FIX: no RandomResizedCrop, no geometric augmentations in transform.
        # Geometric ops (flip/rotate) are done in the Dataset with mask sync.
        # RRCrop is disabled because:
        #   - concat: mask is returned as a separate channel; crop is not
        #     applied to it -> misalignment.
        #   - spatial modes: mask is burned into PIL before transform, so
        #     RRCrop would work, but disabling it here keeps things simple
        #     and consistent. If random crop is desired, implement it in
        #     the Dataset alongside mask crop sync.
        aug_layers.append(transforms.Resize((img_size, img_size)))
    else:
        # Original behavior: RRCrop + geometric augmentations in transform.
        aug_layers.append(transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)))
        if hflip:
            aug_layers.append(transforms.RandomHorizontalFlip(p=0.5))
        if vflip:
            aug_layers.append(transforms.RandomVerticalFlip(p=0.5))
        if rotation_deg > 0:
            aug_layers.append(transforms.RandomRotation(degrees=rotation_deg))

    if use_color_jitter:
        aug_layers.append(transforms.ColorJitter(
            brightness=color_jitter_brightness,
            contrast=color_jitter_contrast,
            saturation=color_jitter_saturation,
            hue=color_jitter_hue,
        ))

    aug_layers += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]

    train_tf = transforms.Compose(aug_layers)

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_tf, val_tf