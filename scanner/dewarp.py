"""UVDoc geometric dewarp.

Wraps the vendored UVDoc (MIT) network. The model predicts a 2D sampling
grid from a downscaled copy of the photo, then that grid remaps the
full-resolution original — the document is flattened by moving existing
pixels, never regenerating them. CPU by default; uses CUDA if present.

The model is loaded lazily and cached, so importing this module is cheap
and the ~32 MB weights only load on the first scan.
"""
from __future__ import annotations

import os
import sys
import threading

import cv2
import numpy as np

# Vendored UVDoc lives next to this package; add it to the path so its
# bare imports (`from model import ...`) resolve to its own files.
_UVDOC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uvdoc")
if _UVDOC_DIR not in sys.path:
    sys.path.insert(0, _UVDOC_DIR)

_CKPT = os.path.join(_UVDOC_DIR, "best_model.pkl")

_model = None
_device = None
_lock = threading.Lock()


def _ensure_model():
    """Load + cache the UVDoc model (thread-safe, lazy)."""
    global _model, _device
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        import torch  # lazy: keeps cold-start fast
        from utils import load_model  # from vendored uvdoc/

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = load_model(_CKPT)
        m.to(_device)
        m.eval()
        _model = m


def dewarp(image_bgr: np.ndarray) -> np.ndarray:
    """Geometrically unwarp a document photo.

    Args:
        image_bgr: HxWx3 uint8 BGR image.
    Returns:
        HxWx3 uint8 BGR unwarped image (same resolution as the input).
    """
    import torch
    from utils import IMG_SIZE, bilinear_unwarping

    _ensure_model()

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = torch.from_numpy(cv2.resize(rgb, IMG_SIZE).transpose(2, 0, 1)).unsqueeze(0).to(_device)

    with torch.no_grad():
        point_positions2D, _ = _model(inp)

    size = rgb.shape[:2][::-1]  # (w, h)
    unwarped = bilinear_unwarping(
        warped_img=torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(_device),
        point_positions=torch.unsqueeze(point_positions2D[0], dim=0),
        img_size=tuple(size),
    )
    out = (unwarped[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
