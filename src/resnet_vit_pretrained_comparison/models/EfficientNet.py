# resnet_vit_pretrained_comparison/models/EfficientNet.py
# EfficientNet with Meta features + optional fused-mask concat support
# + optional mask-guided spatial attention / context-branch support

from __future__ import annotations

import warnings
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# Backbone builder
# ============================================================

def build_efficientnet_backbone(
    arch: str = "efficientnet_b0",
    pretrained: bool = True,
    in_channels: int = 3,
) -> Tuple[nn.Module, int]:
    """
    Build EfficientNet backbone with classifier removed.

    Supports:
        in_channels=3 -> regular RGB input
        in_channels=4 -> RGB + mask channel, for old concat-mask experiments

    Returns:
        backbone:
            Torchvision EfficientNet model with classifier replaced by Identity.
            If called directly as backbone(images), it returns the usual pooled
            image embedding, exactly as before.

        img_emb_dim:
            EfficientNet image embedding dimension.
    """
    arch = str(arch).strip().lower()

    if arch == "efficientnet_b0":
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            model = models.efficientnet_b0(weights=weights)
        else:
            model = models.efficientnet_b0(weights=None)
        img_emb_dim = 1280

    elif arch == "efficientnet_b1":
        if pretrained:
            weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1
            model = models.efficientnet_b1(weights=weights)
        else:
            model = models.efficientnet_b1(weights=None)
        img_emb_dim = 1280

    elif arch == "efficientnet_b2":
        if pretrained:
            weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1
            model = models.efficientnet_b2(weights=weights)
        else:
            model = models.efficientnet_b2(weights=None)
        img_emb_dim = 1408

    else:
        raise ValueError(
            f"Unknown arch: {arch}. "
            f"Options: efficientnet_b0, efficientnet_b1, efficientnet_b2"
        )

    in_channels = int(in_channels)

    # ------------------------------------------------------------
    # Optional 4-channel input support for old mask_mode='concat'
    # ------------------------------------------------------------
    if in_channels == 4:
        original_conv = model.features[0][0]
        original_weight = original_conv.weight.detach().clone()

        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=(original_conv.bias is not None),
        )

        with torch.no_grad():
            # First 3 channels inherit ImageNet RGB weights.
            new_conv.weight[:, :3, :, :].copy_(original_weight)

            # 4th mask channel is initialized as mean of RGB filters.
            new_conv.weight[:, 3:4, :, :].copy_(
                original_weight.mean(dim=1, keepdim=True)
            )

            if original_conv.bias is not None:
                new_conv.bias.copy_(original_conv.bias.detach())

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

    # Remove original EfficientNet classifier.
    # Torchvision EfficientNet forward then returns the pooled embedding.
    model.classifier = nn.Identity()

    return model, img_emb_dim


# ============================================================
# EfficientNet + Metadata
# ============================================================

class EfficientNetWithMeta(nn.Module):
    """
    EfficientNet classifier with optional metadata branch.

    Backward-compatible behavior:
        If use_mask_attention=False, image features are computed exactly with:

            img_feat = self.backbone(images.float())

        This preserves the old execution path.

    Optional mask-guided spatial attention:
        If use_mask_attention=True, forward/forward_features can receive:

            attention_mask: Tensor of shape (B,1,H,W) or (B,H,W)

        The mask is resized to the final EfficientNet feature-map resolution and
        softly boosts feature-map locations inside the mask:

            feature_map = feature_map * (1 + alpha * resized_mask)

        This is intentionally a soft attention mechanism. It does not zero out
        the background, so imperfect masks are less likely to destroy useful
        information.

    Optional mask-context branch:
        If use_mask_context_branch=True, forward/forward_features can also receive:

            mask_context_images: Tensor of shape (B,3,H,W)

        These are intended to be full-image tensors where the lesion remains visible
        and the background is suppressed/blurred by the fused mask. A second
        EfficientNet encoder extracts context features and concatenates them with
        the crop/original image features before metadata and classification.
    """

    def __init__(
        self,
        num_classes: int,
        arch: str = "efficientnet_b0",
        pretrained: bool = True,
        meta_feature_dim: int = 0,
        use_meta: bool = True,
        meta_hidden_dim: int = 128,
        clf_hidden_dim: int = 256,
        dropout: float = 0.3,
        in_channels: int = 3,
        use_mask_attention: bool = False,
        mask_attention_alpha: float = 0.2,
        mask_attention_base_weight: float = 1.0,
        use_mask_context_branch: bool = False,
        mask_context_in_channels: int = 3,
        mask_context_branch_mode: str = "independent",
    ):
        super().__init__()

        self.arch = str(arch).strip().lower()
        self.use_meta = bool(use_meta)
        self.in_channels = int(in_channels)

        self.use_mask_attention = bool(use_mask_attention)
        self.mask_attention_alpha = float(mask_attention_alpha)
        self.mask_attention_base_weight = float(mask_attention_base_weight)
        self.use_mask_context_branch = bool(use_mask_context_branch)
        self.mask_context_in_channels = int(mask_context_in_channels)
        self.mask_context_branch_mode = str(
            mask_context_branch_mode or "independent"
        ).strip().lower()
        if self.mask_context_branch_mode not in {"independent", "shared"}:
            raise ValueError(
                "mask_context_branch_mode must be 'independent' or 'shared', "
                f"got {self.mask_context_branch_mode!r}"
            )

        if self.mask_attention_alpha < 0.0:
            raise ValueError(
                f"mask_attention_alpha must be >= 0, got {self.mask_attention_alpha}"
            )
        if self.mask_attention_base_weight < 0.0:
            raise ValueError(
                "mask_attention_base_weight must be >= 0, "
                f"got {self.mask_attention_base_weight}"
            )

        self.backbone, img_emb_dim = build_efficientnet_backbone(
            arch=self.arch,
            pretrained=bool(pretrained),
            in_channels=self.in_channels,
        )
        self.img_emb_dim = int(img_emb_dim)

        if self.use_mask_context_branch and self.mask_context_branch_mode == "independent":
            if self.mask_context_in_channels != 3:
                raise ValueError(
                    "mask_context_in_channels currently supports RGB tensors only. "
                    f"Got {self.mask_context_in_channels}."
                )

            self.mask_context_backbone, mask_context_emb_dim = build_efficientnet_backbone(
                arch=self.arch,
                pretrained=bool(pretrained),
                in_channels=self.mask_context_in_channels,
            )
            self.mask_context_emb_dim = int(mask_context_emb_dim)
        elif self.use_mask_context_branch and self.mask_context_branch_mode == "shared":
            if self.mask_context_in_channels != self.in_channels:
                raise ValueError(
                    "mask_context_branch_mode='shared' requires the context image "
                    f"channels to match the main branch. Got context={self.mask_context_in_channels}, "
                    f"main={self.in_channels}."
                )
            self.mask_context_backbone = None
            self.mask_context_emb_dim = int(img_emb_dim)
        else:
            self.mask_context_backbone = None
            self.mask_context_emb_dim = 0

        clf_in_dim = int(img_emb_dim) + int(self.mask_context_emb_dim)

        if self.use_meta:
            meta_feature_dim = int(meta_feature_dim)
            if meta_feature_dim <= 0:
                raise ValueError(
                    f"use_meta=True but meta_feature_dim={meta_feature_dim}. "
                    f"Expected meta_feature_dim > 0."
                )

            self.meta_net = nn.Sequential(
                nn.Linear(meta_feature_dim, meta_hidden_dim),
                nn.BatchNorm1d(meta_hidden_dim),
                nn.ReLU(inplace=True),
            )
            clf_in_dim += int(meta_hidden_dim)
        else:
            self.meta_net = None

        self.classifier = nn.Sequential(
            nn.Linear(clf_in_dim, clf_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(clf_hidden_dim, num_classes),
        )

    def _validate_images(self, images: torch.Tensor) -> None:
        if images is None:
            raise ValueError("images is None")

        if images.ndim != 4:
            raise ValueError(
                f"Expected images shape (B,C,H,W), got {tuple(images.shape)}"
            )

        if images.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels}-channel input, got {images.shape[1]}. "
                f"If using mask_mode='concat', create model with in_channels=4. "
                f"For crop-only EfficientNet, in_channels should be 3."
            )

    def _validate_attention_mask(
        self,
        attention_mask: torch.Tensor | None,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        if attention_mask is None:
            raise ValueError(
                "use_mask_attention=True but attention_mask is None. "
                "Pass attention_mask or set use_mask_attention=False."
            )

        if not torch.is_tensor(attention_mask):
            raise TypeError(
                f"attention_mask must be a torch.Tensor, got {type(attention_mask)}"
            )

        if attention_mask.ndim == 3:
            attention_mask = attention_mask.unsqueeze(1)

        if attention_mask.ndim != 4:
            raise ValueError(
                "Expected attention_mask shape (B,1,H,W) or (B,H,W), "
                f"got {tuple(attention_mask.shape)}"
            )

        if attention_mask.shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch: images batch={batch_size}, "
                f"attention_mask batch={attention_mask.shape[0]}"
            )

        if attention_mask.shape[1] != 1:
            raise ValueError(
                f"attention_mask must have one channel, got shape={tuple(attention_mask.shape)}"
            )

        return attention_mask.float().clamp(0.0, 1.0)

    def _build_image_features(
        self,
        images: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        apply_mask_attention: bool = True,
    ) -> torch.Tensor:
        """
        Build EfficientNet image embedding.

        Critical backward-compatibility rule:
            If use_mask_attention=False, this function calls self.backbone(images)
            exactly as the old code did.
        """
        self._validate_images(images)
        images = images.float()

        if not (self.use_mask_attention and bool(apply_mask_attention)):
            return self.backbone(images)

        attention_mask = self._validate_attention_mask(
            attention_mask,
            batch_size=images.shape[0],
        ).to(device=images.device, dtype=images.dtype)

        # Torchvision EfficientNet internals:
        #   features -> avgpool -> flatten -> classifier
        # The classifier was replaced with Identity, so for attention we manually
        # apply the same path but insert the mask before avgpool.
        feature_map = self.backbone.features(images)

        mask_small = F.interpolate(
            attention_mask,
            size=feature_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)

        # Soft feature-map modulation, not hard masking:
        #   outside mask => mask_attention_base_weight
        #   inside mask  => 1 + mask_attention_alpha
        # Default base=1 keeps old boost-only behavior:
        #   outside=1, inside=1+alpha.
        inside_weight = 1.0 + self.mask_attention_alpha
        outside_weight = self.mask_attention_base_weight
        if inside_weight != 1.0 or outside_weight != 1.0:
            multiplier = outside_weight + (inside_weight - outside_weight) * mask_small
            feature_map = feature_map * multiplier

        pooled = self.backbone.avgpool(feature_map)
        img_feat = torch.flatten(pooled, 1)

        return img_feat

    def _validate_mask_context_images(
        self,
        mask_context_images: torch.Tensor | None,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        if mask_context_images is None:
            raise ValueError(
                "use_mask_context_branch=True but mask_context_images is None. "
                "Pass mask_context_images or set use_mask_context_branch=False."
            )

        if not torch.is_tensor(mask_context_images):
            raise TypeError(
                "mask_context_images must be a torch.Tensor, "
                f"got {type(mask_context_images)}"
            )

        if mask_context_images.ndim != 4:
            raise ValueError(
                "Expected mask_context_images shape (B,3,H,W), "
                f"got {tuple(mask_context_images.shape)}"
            )

        if mask_context_images.shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch: images batch={batch_size}, "
                f"mask_context_images batch={mask_context_images.shape[0]}"
            )

        if mask_context_images.shape[1] != self.mask_context_in_channels:
            raise ValueError(
                f"Expected mask_context_images with {self.mask_context_in_channels} channels, "
                f"got {mask_context_images.shape[1]}."
            )

        return mask_context_images.float()

    def _build_mask_context_features(
        self,
        mask_context_images: torch.Tensor | None,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        if not self.use_mask_context_branch:
            if mask_context_images is not None:
                raise ValueError(
                    "mask_context_images was passed, but use_mask_context_branch=False."
                )
            raise RuntimeError("_build_mask_context_features called while branch is disabled.")

        context_images = self._validate_mask_context_images(
            mask_context_images,
            batch_size=batch_size,
        )
        if self.mask_context_branch_mode == "shared":
            return self._build_image_features(
                context_images,
                attention_mask=None,
                apply_mask_attention=False,
            )

        return self.mask_context_backbone(context_images)

    def _build_combined_features(
        self,
        images: torch.Tensor,
        meta_vecs: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        mask_context_images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Build the feature vector that goes into self.classifier.

        Returns:
            combined_features:
                if use_meta=False: image features only, shape (B,img_emb_dim)
                if use_meta=True: concat(image features, metadata features)
        """
        img_feat = self._build_image_features(
            images=images,
            attention_mask=attention_mask,
        )

        feature_parts = [img_feat]

        if self.use_mask_context_branch:
            context_feat = self._build_mask_context_features(
                mask_context_images=mask_context_images,
                batch_size=images.shape[0],
            )
            feature_parts.append(context_feat)
        elif mask_context_images is not None:
            raise ValueError(
                "mask_context_images was passed, but use_mask_context_branch=False."
            )

        if not self.use_meta:
            return torch.cat(feature_parts, dim=1)

        if meta_vecs is None:
            raise ValueError(
                "use_meta=True but meta_vecs is None. "
                "Pass meta_vecs or set use_meta=False."
            )

        if meta_vecs.ndim != 2:
            raise ValueError(
                f"Expected meta_vecs shape (B,F), got {tuple(meta_vecs.shape)}"
            )

        if meta_vecs.shape[0] != images.shape[0]:
            raise ValueError(
                f"Batch size mismatch: images batch={images.shape[0]}, "
                f"meta_vecs batch={meta_vecs.shape[0]}"
            )

        meta_feat = self.meta_net(meta_vecs.float())
        feature_parts.append(meta_feat)
        combined_feat = torch.cat(feature_parts, dim=1)
        return combined_feat

    def forward(
        self,
        images: torch.Tensor,
        meta_vecs: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        mask_context_images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            images:
                Tensor of shape (B,C,H,W), where:
                C=3 for RGB / crop-only RGB
                C=4 for old RGB+mask concat mode

            meta_vecs:
                Tensor of shape (B,meta_feature_dim), if use_meta=True.

            attention_mask:
                Optional tensor of shape (B,1,H,W) or (B,H,W).
                Used only when use_mask_attention=True.

            mask_context_images:
                Optional tensor of shape (B,3,H,W).
                Required only when use_mask_context_branch=True.

        Returns:
            logits of shape (B,num_classes)
        """
        combined_feat = self._build_combined_features(
            images=images,
            meta_vecs=meta_vecs,
            attention_mask=attention_mask,
            mask_context_images=mask_context_images,
        )
        logits = self.classifier(combined_feat)
        return logits

    def forward_features(
        self,
        images: torch.Tensor,
        meta_vecs: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        mask_context_images: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits:
                shape (B,num_classes)

            combined_features:
                representation right before the final classifier.

        This method is required by feature consistency loss in
        run_efficientnet_with_meta.py.
        """
        combined_feat = self._build_combined_features(
            images=images,
            meta_vecs=meta_vecs,
            attention_mask=attention_mask,
            mask_context_images=mask_context_images,
        )
        logits = self.classifier(combined_feat)
        return logits, combined_feat


# ============================================================
# EfficientNet image-only constructor
# ============================================================

def EfficientNetOnly(
    num_classes: int,
    arch: str = "efficientnet_b0",
    pretrained: bool = True,
    dropout: float = 0.3,
    in_channels: int = 3,
    use_mask_attention: bool = False,
    mask_attention_alpha: float = 0.2,
    mask_attention_base_weight: float = 1.0,
    use_mask_context_branch: bool = False,
    mask_context_branch_mode: str = "independent",
) -> EfficientNetWithMeta:
    """
    Convenience constructor:
    EfficientNetWithMeta with use_meta=False.
    """
    return EfficientNetWithMeta(
        num_classes=num_classes,
        arch=arch,
        pretrained=pretrained,
        meta_feature_dim=0,
        use_meta=False,
        dropout=dropout,
        in_channels=in_channels,
        use_mask_attention=use_mask_attention,
        mask_attention_alpha=mask_attention_alpha,
        mask_attention_base_weight=mask_attention_base_weight,
        use_mask_context_branch=use_mask_context_branch,
        mask_context_branch_mode=mask_context_branch_mode,
    )


# ============================================================
# Transforms
# ============================================================

def efficientnet_transforms(
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
    Build train/eval transforms for regular EfficientNet.

    Important:
        Random spatial augmentations are intentionally NOT applied here.

    Why:
        In mask/crop experiments, geometric augmentations must be applied inside
        the Dataset so image and mask stay synchronized.

    Therefore this transform handles only:
        - resize
        - optional ColorJitter
        - ToTensor
        - Normalize

    The args rotation_deg / hflip / vflip are kept for backward compatibility,
    but are intentionally ignored here.
    """
    from torchvision import transforms

    img_size = int(img_size)

    if pretrained:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    train_augs = [
        transforms.Resize((img_size, img_size)),
    ]

    if use_color_jitter and (
        float(color_jitter_brightness) > 0
        or float(color_jitter_contrast) > 0
        or float(color_jitter_saturation) > 0
        or float(color_jitter_hue) > 0
    ):
        train_augs.append(
            transforms.ColorJitter(
                brightness=float(color_jitter_brightness),
                contrast=float(color_jitter_contrast),
                saturation=float(color_jitter_saturation),
                hue=float(color_jitter_hue),
            )
        )

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


__all__ = [
    "build_efficientnet_backbone",
    "EfficientNetWithMeta",
    "EfficientNetOnly",
    "efficientnet_transforms",
]
