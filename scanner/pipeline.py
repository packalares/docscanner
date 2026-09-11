"""End-to-end scan: dewarp -> cleanup, driven by Config (env + overrides)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import dewarp as _dewarp
from . import enhance as _enhance
from .config import Config


def scan(image_bgr: np.ndarray, cfg: Optional[Config] = None, **overrides) -> np.ndarray:
    """Run the full pipeline on one BGR image.

    Args:
        image_bgr: HxWx3 uint8 BGR.
        cfg: base Config; defaults to Config.from_env().
        **overrides: Config field names (mode, do_dewarp, sharpen,
            white_point, ...) — non-None values override the base config.
    Returns:
        HxWx3 uint8 BGR cleaned page.
    """
    cfg = (cfg or Config.from_env()).override(**overrides)
    img = _dewarp.dewarp(image_bgr) if cfg.do_dewarp else image_bgr
    return _enhance.render(img, cfg)
