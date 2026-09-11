# docscanner

Phone photo → clean, dewarped scan. CPU-only, ~1–2 s/page, stateless
(in-memory; nothing written to disk). Pixels are never regenerated —
UVDoc remaps the original photo through a learned deformation field, then
classical OpenCV steps normalize lighting and contrast.

```
photo → UVDoc dewarp → flatten → CLAHE → unsharp → levels → edge-protected whitening
```

## API

| Method | Path | In | Out |
|---|---|---|---|
| GET | `/health` | — | `{"status":"ok"}` |
| GET | `/config` | — | effective env config |
| POST | `/scan` | multipart `file` | cleaned image bytes (`image/png`) |
| POST | `/scan/json` | `{"image_b64",…}` | `{"image_b64","format"}` |
| POST | `/scan/pdf` | multipart `files` | combined `application/pdf` |

Per-request overrides (query params / body fields): `mode`, `dewarp`,
`sharpen`, `white_point`, `format`. Anything unset uses the env default.

## Configuration (env — `DOCSCANNER_*`)

Every step is toggleable and every strength tunable without a rebuild:

| Var | Default | Meaning |
|---|---|---|
| `MODE` | `clean` | `clean` \| `natural` \| `grayscale` \| `binary` |
| `DEWARP` | `1` | run UVDoc geometric correction |
| `FLATTEN` | `1` | illumination flatten |
| `CLAHE` | `1` | local-contrast step |
| `CLAHE_CLIP` | `1.3` | CLAHE clip limit |
| `SHARPEN` | `1.6` | unsharp amount (`0` = off) |
| `LEVELS` | `1` | black/white-point stretch |
| `BLACK_POINT` | `45` | levels black point |
| `WHITE_POINT` | `212` | levels white point (lower = whiter) |
| `WHITEN` | `1` | edge-protected stain removal |
| `FORMAT` | `png` | `png` \| `jpg` |

## Run locally

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app:app --port 5002
```

## Docker

```bash
docker build -t docscanner .
docker run -p 5002:5002 docscanner
```

CI builds and pushes `ghcr.io/packalares/docscanner:latest` on push to `main`.

## Third-party

Bundles **UVDoc** (`uvdoc/`, MIT — tanguymagne/UVDoc) including its 32 MB
model weights, so there are no runtime downloads.
