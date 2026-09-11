"""docscanner HTTP API.

Phone photo -> clean, dewarped scan. Stateless: images are processed in
memory and returned in the response — nothing is written to disk.

Endpoints:
  GET  /health          -> {"status": "ok"}
  GET  /config          -> the effective env config (defaults)
  POST /scan            multipart file  -> cleaned image bytes (image/png|jpeg)
  POST /scan/json       JSON base64     -> {"image_b64", "format"}
  POST /scan/pdf        multipart files -> a single combined PDF (application/pdf)

Defaults come from DOCSCANNER_* env vars (see scanner.config). Any query
param / body field provided on a request overrides the env default for that
request; anything left unset uses the env value.
"""
from __future__ import annotations

import base64
from dataclasses import asdict
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from scanner.config import Config
from scanner.export import encode_image, encode_pdf
from scanner.pipeline import scan

app = FastAPI(title="docscanner", version="1.0.0")


def _decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    return img


def _media_type(fmt: str) -> str:
    return "image/jpeg" if fmt.lower().lstrip(".") in ("jpg", "jpeg") else "image/png"


def _overrides(mode, dewarp, sharpen, white_point) -> dict:
    """Build a Config-override dict from optional request params (skip None)."""
    ov = {}
    if mode is not None:
        ov["mode"] = mode
    if dewarp is not None:
        ov["do_dewarp"] = dewarp
    if sharpen is not None:
        ov["sharpen"] = sharpen
    if white_point is not None:
        ov["white_point"] = white_point
    return ov


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/config")
def config():
    """The effective env-derived defaults (useful for debugging)."""
    return asdict(Config.from_env())


@app.post("/scan")
async def scan_endpoint(
    file: UploadFile = File(...),
    mode: Optional[str] = Query(None),
    dewarp: Optional[bool] = Query(None),
    format: Optional[str] = Query(None),
    sharpen: Optional[float] = Query(None),
    white_point: Optional[float] = Query(None),
):
    """Multipart upload -> cleaned image bytes."""
    img = _decode(await file.read())
    out = scan(img, **_overrides(mode, dewarp, sharpen, white_point))
    fmt = format or Config.from_env().fmt
    return Response(content=encode_image(out, fmt), media_type=_media_type(fmt))


class ScanJson(BaseModel):
    image_b64: str
    mode: Optional[str] = None
    dewarp: Optional[bool] = None
    format: Optional[str] = None
    sharpen: Optional[float] = None
    white_point: Optional[float] = None


@app.post("/scan/json")
def scan_json(body: ScanJson):
    """Base64 JSON in -> base64 JSON out (convenient for service-to-service)."""
    raw = base64.b64decode(body.image_b64.split(",")[-1])
    img = _decode(raw)
    out = scan(img, **_overrides(body.mode, body.dewarp, body.sharpen, body.white_point))
    fmt = body.format or Config.from_env().fmt
    b64 = base64.b64encode(encode_image(out, fmt)).decode("ascii")
    return JSONResponse({"image_b64": b64, "format": fmt.lower().lstrip(".")})


@app.post("/scan/pdf")
async def scan_pdf(
    files: List[UploadFile] = File(...),
    mode: Optional[str] = Query(None),
    dewarp: Optional[bool] = Query(None),
    sharpen: Optional[float] = Query(None),
    white_point: Optional[float] = Query(None),
):
    """Multipart upload of one+ images -> a single combined PDF."""
    ov = _overrides(mode, dewarp, sharpen, white_point)
    pages = []
    for f in files:
        img = _decode(await f.read())
        pages.append(scan(img, **ov))
    return Response(content=encode_pdf(pages), media_type="application/pdf")
