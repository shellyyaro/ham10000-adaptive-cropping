# src/training/common_imbalance.py
from __future__ import annotations

import warnings
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Weights / sampler helpers
# -------------------------
def weights_inverse_freq(counts: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    inv = 1.0 / np.maximum(counts, eps)
    w = inv / inv.mean()
    return w


def weights_inverse_sqrt_freq(counts: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    counts = np.maximum(counts, eps)
    inv_sqrt = 1.0 / np.sqrt(counts)
    w = inv_sqrt / inv_sqrt.mean()
    return w


def weights_effective_num(counts: np.ndarray, beta: float = 0.9999, eps: float = 1e-12) -> np.ndarray:
    counts = np.maximum(counts, 0.0)
    eff = (1.0 - np.power(beta, counts))
    w = (1.0 - beta) / np.maximum(eff, eps)
    w = w / w.mean()
    return w


# -------------------------
# Logit transforms
# -------------------------
def apply_logit_adjustment(logits: torch.Tensor, priors: torch.Tensor, tau: float) -> torch.Tensor:
    """Post-hoc logit adjustment (Menon et al., 2021): subtract tau*log(priors) at inference."""
    return logits - tau * torch.log(priors)


def apply_balanced_softmax(logits: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Balanced Softmax (Ren et al., 2020): logits + log(counts) so that
    softmax(z + log(n))_y = n_y * exp(z_y) / sum_c n_c * exp(z_c)."""
    return logits + torch.log(counts)


# -------------------------
# Focal loss
# -------------------------
class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = float(gamma)
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        B, C = logits.shape

        if self.label_smoothing > 0:
            with torch.no_grad():
                true_dist = torch.zeros_like(logits).fill_(self.label_smoothing / (C - 1))
                true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
            p_t = (true_dist * probs).sum(dim=1)
            log_p_t = (true_dist * log_probs).sum(dim=1)
        else:
            p_t = probs.gather(1, target.unsqueeze(1)).squeeze(1)
            log_p_t = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)

        focal_factor = (1.0 - p_t).clamp(min=0.0) ** self.gamma
        loss = -focal_factor * log_p_t

        if self.alpha is not None:
            at = self.alpha.gather(0, target)
            loss = loss * at

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# -------------------------
# Shared builder: train_logits_tf + eval_logits_tf + loss_fn
# -------------------------
def build_loss_and_logits_transform(
    *,
    loss_name: str,
    class_weighting: str,
    label_smoothing: float,
    focal_gamma: float,
    focal_alpha_mode: str,
    logit_adjustment: str,
    tau: float,
    class_counts_np: np.ndarray,
    device: torch.device,
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],   # train_logits_tf
    Callable[[torch.Tensor], torch.Tensor],   # eval_logits_tf
    nn.Module,                                # loss_fn
]:
    """
    Returns:
      train_logits_tf(logits) -> logits for training forward pass
      eval_logits_tf(logits)  -> logits for inference / evaluation
      loss_fn(logits_transformed, targets) -> loss

    Logit-adjustment modes:
      - "none"                    : identity at both train and eval
      - "logit_adjust"            : identity at train, subtract tau*log(priors) at eval  (Menon et al.)
      - "balanced_softmax"        : add log(counts) at train, identity at eval
      - "balanced_softmax_both"   : add log(counts) at both train and eval

    Why balanced_softmax uses identity at eval:
      Balanced Softmax modifies the CE loss during training so the model learns
      features that are not biased by class frequency. At inference, the model's
      raw logits already reflect unbiased predictions. Applying z + log(counts) at
      eval re-introduces majority-class bias: argmax(z + log(n)) favors classes
      with large n, which suppresses recall for rare classes like melanoma.
    """
    counts = torch.tensor(class_counts_np, dtype=torch.float32, device=device)
    counts = torch.clamp(counts, min=1.0)
    priors = counts / counts.sum()

    # ---- class weights (CE weight or focal alpha) ----
    class_weighting = (class_weighting or "none").lower()
    weight_t = None
    if class_weighting == "inverse_freq":
        w = weights_inverse_freq(class_counts_np)
        weight_t = torch.tensor(w, dtype=torch.float32, device=device)
    elif class_weighting == "inverse_sqrt_freq":
        w = weights_inverse_sqrt_freq(class_counts_np)
        weight_t = torch.tensor(w, dtype=torch.float32, device=device)
    elif class_weighting == "effective_num":
        w = weights_effective_num(class_counts_np, beta=0.9999)
        weight_t = torch.tensor(w, dtype=torch.float32, device=device)
    elif class_weighting in ("none", "null", ""):
        weight_t = None
    else:
        raise ValueError(f"Unknown class_weighting: {class_weighting}")

    # ---- logits transforms (train vs eval) ----
    logit_adjustment = (logit_adjustment or "none").lower()

    if class_weighting not in ("none", "null", "") and logit_adjustment != "none":
        warnings.warn(
            f"Using class_weighting='{class_weighting}' together with "
            f"logit_adjustment='{logit_adjustment}' may cause double compensation "
            f"for class imbalance. Consider using only one of these strategies.",
            UserWarning,
            stacklevel=2,
        )

    if logit_adjustment == "none":
        def train_logits_tf(z):
            return z

        def eval_logits_tf(z):
            return z

    elif logit_adjustment == "logit_adjust":
        tau_f = float(tau)

        def train_logits_tf(z):
            return z

        def eval_logits_tf(z):
            return apply_logit_adjustment(z, priors, tau=tau_f)

    elif logit_adjustment == "balanced_softmax":
        def train_logits_tf(z):
            return apply_balanced_softmax(z, counts)

        def eval_logits_tf(z):
            return z

    elif logit_adjustment == "balanced_softmax_both":
        warnings.warn(
            "balanced_softmax_both applies z + log(counts) at eval, which biases "
            "predictions toward majority classes. Consider 'balanced_softmax' "
            "(train-only) or 'logit_adjust' for better rare-class recall.",
            UserWarning,
            stacklevel=2,
        )

        def train_logits_tf(z):
            return apply_balanced_softmax(z, counts)

        def eval_logits_tf(z):
            return apply_balanced_softmax(z, counts)

    else:
        raise ValueError(f"Unknown logit_adjustment: {logit_adjustment}")

    # ---- loss ----
    loss_name = (loss_name or "ce").lower()
    if loss_name == "ce":
        loss_fn = nn.CrossEntropyLoss(
            weight=weight_t,
            label_smoothing=float(label_smoothing),
        )
    elif loss_name == "focal":
        focal_alpha_mode = (focal_alpha_mode or "none").lower()
        alpha_t = weight_t if focal_alpha_mode == "use_class_weights" else None
        loss_fn = FocalLoss(
            gamma=float(focal_gamma),
            alpha=alpha_t,
            label_smoothing=float(label_smoothing),
            reduction="mean",
        )
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")

    return train_logits_tf, eval_logits_tf, loss_fn