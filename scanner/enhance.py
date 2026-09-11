"""Document cleanup — illumination flatten, contrast, sharpen, stain removal.

Pure OpenCV/NumPy (CPU, ~1 s/page). No pixels are invented: every step is a
per-pixel tone/contrast operation on the dewarped photo.

Pipeline (the "clean" mode), in order:
  1. flatten     — divide each channel by a large Gaussian blur of itself,
                   removing uneven lighting / shadow gradients.
  2. clahe       — mild local contrast on the L channel.
  3. unsharp     — sharpen text edges.
  4. levels      — black/white-point stretch (keeps light text, darkens ink).
  5. whiten      — edge-protected: force smooth, bright background to pure
                   white while leaving anything with edges (all text) intact,
                   which erases blotches/stains without wiping faint text.

`grayscale` returns the luminance of `clean`; `binary` runs an adaptive
threshold on it for a hard black-on-white scan; `natural` skips levels +
whitening for a softer paper look.
"""
from __future__ import annotations

import cv2
import numpy as np

MODE_CLEAN = "clean"
MODE_NATURAL = "natural"
MODE_GRAY = "grayscale"
MODE_BINARY = "binary"
ALL_MODES = (MODE_CLEAN, MODE_NATURAL, MODE_GRAY, MODE_BINARY)


def _flatten(image_bgr: np.ndarray) -> np.ndarray:
    """Even out lighting: divide each channel by its blurred self."""
    h, w = image_bgr.shape[:2]
    sigma = max(11.0, min(h, w) / 18.0)
    chans = []
    for c in cv2.split(image_bgr):
        bg = cv2.GaussianBlur(c, (0, 0), sigmaX=sigma, sigmaY=sigma)
        chans.append(cv2.divide(c, bg, scale=255))
    return cv2.merge(chans)


def _edge_protected_whiten(L: np.ndarray, L_for_edges: np.ndarray,
                           bright: float = 175.0) -> np.ndarray:
    """Force smooth, bright background to white; protect anything on an edge.

    `L` is the (levels-adjusted) luminance to modify; `L_for_edges` is the
    luminance used to find text edges (so text — even light grey — is kept).
    """
    lap = cv2.Laplacian(L_for_edges, cv2.CV_32F, ksize=3)
    content = (np.abs(lap) > 5).astype(np.uint8)
    content = cv2.dilate(
        content, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1
    ).astype(bool)
    out = L.copy()
    out[(~content) & (L > bright)] = 255.0
    return out


def clean(
    image_bgr: np.ndarray,
    *,
    do_flatten: bool = True,
    do_clahe: bool = True,
    clahe_clip: float = 1.3,
    sharpen: float = 1.6,
    do_levels: bool = True,
    black_point: float = 45.0,
    white_point: float = 212.0,
    whiten: bool = True,
) -> np.ndarray:
    """Full color cleanup. Returns HxWx3 uint8 BGR.

    Each step is individually toggleable. `sharpen` is the unsharp amount
    (0 = off). `white_point` lower = whiter background (more aggressive stain
    removal). Turn `whiten`/`do_levels` off for a softer 'natural' look.
    """
    img = _flatten(image_bgr) if do_flatten else image_bgr.copy()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    if do_clahe:
        L = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8)).apply(L)
    bgr = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

    if sharpen and sharpen > 0:
        blur = cv2.GaussianBlur(bgr, (0, 0), 1.4)
        bgr = cv2.addWeighted(bgr, 1.0 + sharpen, blur, -sharpen, 0)

    lab2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L2, A2, B2 = cv2.split(lab2)
    Lc = L2.astype(np.float32)
    if do_levels:
        Lc = np.clip((Lc - black_point) * (255.0 / max(1.0, white_point - black_point)), 0, 255)
    if whiten:
        Lc = _edge_protected_whiten(Lc, L2)
    out = cv2.merge([Lc.astype(np.uint8), A2, B2])
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def grayscale(image_bgr: np.ndarray, **kw) -> np.ndarray:
    """Cleanup then convert to single-channel grey (returned as BGR)."""
    g = cv2.cvtColor(clean(image_bgr, **kw), cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def binary(image_bgr: np.ndarray, block_frac: float = 30.0, offset: int = 12,
           **kw) -> np.ndarray:
    """Hard black-on-white scan via adaptive threshold on the cleaned grey."""
    g = cv2.cvtColor(clean(image_bgr, **kw), cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]
    bs = int(min(h, w) / block_frac) | 1
    th = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, max(11, bs), offset
    )
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def render(image_bgr: np.ndarray, cfg) -> np.ndarray:
    """Dispatch by output mode using a Config (see scanner.config.Config).

    `cfg` is duck-typed — any object exposing the step fields works.
    """
    kw = dict(
        do_flatten=cfg.do_flatten,
        do_clahe=cfg.do_clahe,
        clahe_clip=cfg.clahe_clip,
        sharpen=cfg.sharpen,
        do_levels=cfg.do_levels,
        black_point=cfg.black_point,
        white_point=cfg.white_point,
        whiten=cfg.do_whiten,
    )
    mode = cfg.mode
    if mode == MODE_GRAY:
        return grayscale(image_bgr, **kw)
    if mode == MODE_BINARY:
        return binary(image_bgr, **kw)
    if mode == MODE_NATURAL:
        return clean(image_bgr, **{**kw, "do_levels": False, "whiten": False})
    return clean(image_bgr, **kw)
