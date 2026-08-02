FROM python:3.12-slim

ARG PYTORCH_INSTALL_CUDA=true
ARG PYTORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime dependencies required by the pipeline and build tooling
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git libtk8.6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && if [ "${PYTORCH_INSTALL_CUDA}" = "true" ] && [ -n "${PYTORCH_CUDA_INDEX_URL}" ]; then \
        pip install --no-cache-dir torch==2.8.0+cu128 torchaudio==2.8.0+cu128 torchvision==0.23.0+cu128 --index-url "${PYTORCH_CUDA_INDEX_URL}"; \
    else \
        pip install --no-cache-dir torch==2.8.0+cpu torchaudio==2.8.0+cpu torchvision==0.23.0+cpu --index-url https://download.pytorch.org/whl/cpu; \
    fi \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-bundle the speaker diarization model (~33 MB) from ModelScope so users of
# the image can run `legen --diarize` without an internet connection or token.
# Downloads are skipped if the cache is already valid.
RUN python -c "from diarization_utils import ensure_diarization_model; ensure_diarization_model()"

ENTRYPOINT ["python", "legen.py"]
CMD ["--help"]
