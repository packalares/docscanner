# docscanner — CPU-only document scanner (UVDoc dewarp + OpenCV cleanup).
FROM python:3.12-slim

# libglib2.0-0 is required by opencv-python-headless at import time.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU PyTorch from the dedicated wheel index (keeps the image off CUDA).
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + vendored UVDoc (weights baked in — no runtime downloads).
COPY uvdoc/ ./uvdoc/
COPY scanner/ ./scanner/
COPY app.py .

# Sensible CPU thread cap so one big page doesn't hog every core.
ENV OMP_NUM_THREADS=4 \
    DOCSCANNER_MODE=clean

EXPOSE 5002
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5002"]
