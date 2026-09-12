"""Pipeline configuration from environment variables.

Every step can be toggled and every strength tuned via env, so the exact
recipe we validated is the default but an operator can change it without a
rebuild. Per-request API params override these when provided.

Env vars (all prefixed DOCSCANNER_):
  MODE          clean | natural | grayscale | binary   (default clean)
  DEWARP        1/0    run UVDoc geometric correction   (default 1)
  FLATTEN       1/0    illumination flatten             (default 1)
  CLAHE         1/0    local-contrast step              (default 1)
  CLAHE_CLIP    float  CLAHE clip limit                 (default 1.3)
  SHARPEN       float  unsharp amount, 0 = off          (default 1.6)
  LEVELS        1/0    black/white-point stretch        (default 1)
  BLACK_POINT   float  levels black point               (default 45)
  WHITE_POINT   float  levels white point (lower=whiter)(default 212)
  WHITEN        1/0    edge-protected stain removal     (default 1)
  FORMAT        png | jpg                               (default png)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace

_PREFIX = "DOCSCANNER_"


def _b(name: str, default: bool) -> bool:
    v = os.environ.get(_PREFIX + name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _f(name: str, default: float) -> float:
    v = os.environ.get(_PREFIX + name)
    try:
        return float(v) if v not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _s(name: str, default: str) -> str:
    v = os.environ.get(_PREFIX + name)
    return v if v not in (None, "") else default


@dataclass(frozen=True)
class Config:
    mode: str = "clean"
    do_dewarp: bool = True
    do_flatten: bool = True
    do_clahe: bool = True
    clahe_clip: float = 1.3
    sharpen: float = 1.6
    do_levels: bool = True
    black_point: float = 45.0
    white_point: float = 212.0
    do_whiten: bool = True
    fmt: str = "png"

    @staticmethod
    def from_env() -> "Config":
        return Config(
            mode=_s("MODE", "clean"),
            do_dewarp=_b("DEWARP", True),
            do_flatten=_b("FLATTEN", True),
            do_clahe=_b("CLAHE", True),
            clahe_clip=_f("CLAHE_CLIP", 1.3),
            sharpen=_f("SHARPEN", 1.6),
            do_levels=_b("LEVELS", True),
            black_point=_f("BLACK_POINT", 45.0),
            white_point=_f("WHITE_POINT", 212.0),
            do_whiten=_b("WHITEN", True),
            fmt=_s("FORMAT", "png"),
        )

    def override(self, **kw) -> "Config":
        """Return a copy with any non-None, *known* kwargs applied.

        Unknown keys are ignored (never raises) so a caller passing an
        unexpected field can't crash the service — it just falls back to the
        existing value for anything it doesn't recognize.
        """
        fields = {f.name for f in self.__dataclass_fields__.values()}
        clean = {k: v for k, v in kw.items() if v is not None and k in fields}
        return replace(self, **clean) if clean else self
