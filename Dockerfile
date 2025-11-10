FROM python:3.12-slim

ARG PYTORCH_INSTALL_CUDA=true
ARG PYTORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu121

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime dependencies required by the pipeline and build tooling
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && if [ "${PYTORCH_INSTALL_CUDA}" = "true" ] && [ -n "${PYTORCH_CUDA_INDEX_URL}" ]; then \
        pip install --no-cache-dir --upgrade torch --index-url "${PYTORCH_CUDA_INDEX_URL}"; \
    fi

COPY . .

ENTRYPOINT ["python", "legen.py"]
CMD ["--help"]