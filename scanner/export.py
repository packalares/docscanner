"""Encoding helpers: image bytes + multi-page PDF. All in-memory."""
from __future__ import annotations

import io
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


def encode_image(image_bgr: np.ndarray, fmt: str = "png", quality: int = 92) -> bytes:
    """Encode a BGR image to PNG/JPEG bytes (no disk)."""
    fmt = fmt.lower().lstrip(".")
    ext = ".jpg" if fmt in ("jpg", "jpeg") else ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext == ".jpg" else []
    ok, buf = cv2.imencode(ext, image_bgr, params)
    if not ok:
        raise RuntimeError("image encode failed")
    return buf.tobytes()


def _to_pil(image_bgr: np.ndarray) -> Image.Image:
    if image_bgr.ndim == 2:
        return Image.fromarray(image_bgr, mode="L").convert("RGB")
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), mode="RGB")


def encode_pdf(images_bgr: Iterable[np.ndarray], resolution: float = 200.0) -> bytes:
    """Combine one or more BGR pages into a single PDF (bytes, no disk)."""
    pages = [_to_pil(im) for im in images_bgr]
    if not pages:
        raise ValueError("encode_pdf needs at least one page")
    buf = io.BytesIO()
    pages[0].save(
        buf, format="PDF", save_all=True, append_images=pages[1:], resolution=resolution
    )
    return buf.getvalue()
