# Quantize-as-a-Tool service. Needs an NVIDIA GPU at runtime (docker run --gpus all).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/content/hf-cache \
    MEM_PATH=/content/memory.json \
    SIGNING_KEY_PATH=/content/signing_key.pem

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY quantize_service.py serve.py verify_report.py memory.json ./
RUN mkdir -p /content/artifacts

# Measurements, signing key, model cache and artifacts persist here.
VOLUME /content
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# API_KEY must be supplied at run time (never baked into the image).
CMD ["python", "serve.py"]
