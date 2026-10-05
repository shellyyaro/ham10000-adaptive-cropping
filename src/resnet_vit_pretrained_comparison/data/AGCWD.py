from __future__ import annotations

import random
from typing import Any

import cv2
import numpy as np
from PIL import Image


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert image to uint8 in [0, 255].

    Supports:
    - uint8 input
    - float input in [0, 1]
    - float input in [0, 255]
    """
    if image.dtype == np.uint8:
        return image

    img = image.astype(np.float32)

    if img.size == 0:
        raise ValueError("Empty image received.")

    if img.max() <= 1.0:
        img = img * 255.0

    return np.clip(img, 0, 255).astype(np.uint8)


def agcwd_gray(
    gray: np.ndarray,
    truncated_cdf: bool = False,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Adaptive Gamma Correction with Weighting Distribution for a grayscale image.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image of shape (H, W).
    truncated_cdf : bool
        Whether to use truncated gamma lower bound.
    alpha : float
        Controls aggressiveness of the weighting distribution.
        Lower alpha -> usually more aggressive enhancement.
    eps : float
        Small numeric stability constant.

    Returns
    -------
    np.ndarray
        Enhanced grayscale image, dtype uint8, shape (H, W).
    """
    gray = _to_uint8(gray)

    if gray.ndim != 2:
        raise ValueError("agcwd_gray expects a grayscale image with shape (H, W).")

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().astype(np.float64)
    pdf = hist / (hist.sum() + eps)

    nonzero_pdf = pdf[pdf > 0]
    if len(nonzero_pdf) == 0:
        return gray.copy()

    pdf_min = nonzero_pdf.min()
    pdf_max = nonzero_pdf.max()

    pdf_wd = np.zeros_like(pdf)

    if pdf_max - pdf_min < eps:
        pdf_wd[pdf > 0] = pdf[pdf > 0]
    else:
        nz = pdf > 0
        pdf_wd[nz] = pdf_max * (((pdf[nz] - pdf_min) / (pdf_max - pdf_min + eps)) ** alpha)

    pdf_wd = pdf_wd / (pdf_wd.sum() + eps)
    cdf_wd = np.cumsum(pdf_wd)

    lut = np.zeros(256, dtype=np.uint8)

    if truncated_cdf:
        for i in range(256):
            gamma = max(0.5, 1.0 - cdf_wd[i])
            out = 255.0 * ((i / 255.0) ** gamma)
            lut[i] = np.uint8(np.clip(round(out), 0, 255))
    else:
        for i in range(256):
            gamma = 1.0 - cdf_wd[i]
            out = 255.0 * ((i / 255.0) ** gamma) if i > 0 else 0.0
            lut[i] = np.uint8(np.clip(round(out), 0, 255))

    return cv2.LUT(gray, lut)


def agcwd_color_bgr(
    image_bgr: np.ndarray,
    method: str = "ycrcb",
    truncated_cdf: bool = False,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Apply AGCWD to luminance only on a color BGR image.

    Parameters
    ----------
    image_bgr : np.ndarray
        Color image in BGR format, shape (H, W, 3).
    method : str
        'ycrcb' or 'hsv'
    truncated_cdf : bool
        Whether to use truncated gamma lower bound.
    alpha : float
        Controls enhancement aggressiveness.

    Returns
    -------
    np.ndarray
        Enhanced BGR image, dtype uint8.
    """
    image_bgr = _to_uint8(image_bgr)

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("agcwd_color_bgr expects image shape (H, W, 3).")

    method = method.lower()

    if method == "ycrcb":
        img_ycc = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(img_ycc)
        y_enh = agcwd_gray(y, truncated_cdf=truncated_cdf, alpha=alpha)
        out_ycc = cv2.merge([y_enh, cr, cb])
        out_bgr = cv2.cvtColor(out_ycc, cv2.COLOR_YCrCb2BGR)
        return out_bgr

    if method == "hsv":
        img_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)
        v_enh = agcwd_gray(v, truncated_cdf=truncated_cdf, alpha=alpha)
        out_hsv = cv2.merge([h, s, v_enh])
        out_bgr = cv2.cvtColor(out_hsv, cv2.COLOR_HSV2BGR)
        return out_bgr

    raise ValueError("method must be 'ycrcb' or 'hsv'")


def apply_agcwd_rgb(
    image_rgb: np.ndarray,
    method: str = "ycrcb",
    truncated_cdf: bool = False,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Apply AGCWD to an RGB image and return RGB output.

    This is the most convenient function for PIL / torchvision pipelines,
    because PIL images are usually interpreted as RGB.

    Parameters
    ----------
    image_rgb : np.ndarray
        RGB image of shape (H, W, 3).
    method : str
        'ycrcb' or 'hsv'
    truncated_cdf : bool
        Whether to use truncated gamma lower bound.
    alpha : float
        Controls enhancement aggressiveness.

    Returns
    -------
    np.ndarray
        Enhanced RGB image, dtype uint8.
    """
    image_rgb = _to_uint8(image_rgb)

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("apply_agcwd_rgb expects image shape (H, W, 3).")

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    out_bgr = agcwd_color_bgr(
        image_bgr=image_bgr,
        method=method,
        truncated_cdf=truncated_cdf,
        alpha=alpha,
    )
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return out_rgb


class AGCWDTransform:
    """
    Torchvision-compatible transform.

    Can be inserted inside transforms.Compose([...]) before ToTensor().

    Supports PIL.Image and numpy RGB arrays.
    """

    def __init__(
        self,
        p: float = 1.0,
        method: str = "ycrcb",
        alpha: float = 0.5,
        truncated_cdf: bool = False,
    ):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}")

        self.p = p
        self.method = method
        self.alpha = alpha
        self.truncated_cdf = truncated_cdf

    def __call__(self, img: Any):
        if random.random() > self.p:
            return img

        if isinstance(img, Image.Image):
            img_np = np.array(img)
            out_np = apply_agcwd_rgb(
                image_rgb=img_np,
                method=self.method,
                truncated_cdf=self.truncated_cdf,
                alpha=self.alpha,
            )
            return Image.fromarray(out_np)

        if isinstance(img, np.ndarray):
            return apply_agcwd_rgb(
                image_rgb=img,
                method=self.method,
                truncated_cdf=self.truncated_cdf,
                alpha=self.alpha,
            )

        raise TypeError(
            f"AGCWDTransform supports PIL.Image or numpy.ndarray, got {type(img)}"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"p={self.p}, "
            f"method='{self.method}', "
            f"alpha={self.alpha}, "
            f"truncated_cdf={self.truncated_cdf})"
        )


AGCWD_PRESETS = {
    "none": None,
    "mild": {
        "p": 1.0,
        "method": "ycrcb",
        "alpha": 0.75,
        "truncated_cdf": False,
    },
    "medium": {
        "p": 1.0,
        "method": "ycrcb",
        "alpha": 0.50,
        "truncated_cdf": False,
    },
    "aggressive": {
        "p": 1.0,
        "method": "ycrcb",
        "alpha": 0.25,
        "truncated_cdf": True,
    },
}


def build_agcwd_transform(
    enabled: bool = False,
    p: float = 1.0,
    method: str = "ycrcb",
    alpha: float = 0.5,
    truncated_cdf: bool = False,
):
    """
    Convenience builder for config-driven pipelines.

    Returns:
    - AGCWDTransform(...) if enabled=True
    - None otherwise
    """
    if not enabled:
        return None

    return AGCWDTransform(
        p=p,
        method=method,
        alpha=alpha,
        truncated_cdf=truncated_cdf,
    )


def build_agcwd_transform_from_preset(preset: str):
    """
    Build transform from preset name.

    Supported presets:
    - none
    - mild
    - medium
    - aggressive
    """
    preset = preset.lower()

    if preset not in AGCWD_PRESETS:
        raise ValueError(
            f"Unknown AGCWD preset '{preset}'. "
            f"Supported: {list(AGCWD_PRESETS.keys())}"
        )

    params = AGCWD_PRESETS[preset]
    if params is None:
        return None

    return AGCWDTransform(**params)

def build_agcwd_from_cfg(cfg: dict):
    """
    Build AGCWD transform from experiment config.

    Priority:
    1) agcwd_preset if provided and not "none"
    2) explicit AGCWD parameters
    3) otherwise None

    Supported preset values:
    - "none"
    - "mild"
    - "medium"
    - "aggressive"

    Supported explicit keys:
    - use_agcwd
    - agcwd_p
    - agcwd_method
    - agcwd_alpha
    - agcwd_truncated_cdf
    """
    if cfg is None:
        return None

    preset = str(cfg.get("agcwd_preset", "none")).lower()

    if preset != "none":
        return build_agcwd_transform_from_preset(preset)

    if not bool(cfg.get("use_agcwd", False)):
        return None

    return build_agcwd_transform(
        enabled=True,
        p=float(cfg.get("agcwd_p", 1.0)),
        method=str(cfg.get("agcwd_method", "ycrcb")),
        alpha=float(cfg.get("agcwd_alpha", 0.5)),
        truncated_cdf=bool(cfg.get("agcwd_truncated_cdf", False)),
    )