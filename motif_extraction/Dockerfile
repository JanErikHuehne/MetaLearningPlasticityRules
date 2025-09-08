# CUDA 12.2 runtime + Debian Bookworm (stable)
FROM nvidia/cuda:12.2.2-runtime-bookworm

LABEL org.opencontainers.image.title="Container for Jan - SBI Initial Runs" \
      org.opencontainers.image.description="Run a single SBI script with pinned dependencies" \
      org.opencontainers.image.authors="jan.huehne@tum.de" \
      org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    nas_path=/nas

# System deps: Python, build tools, Fortran, and the CBC solver PuLP will invoke
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv \
      build-essential gfortran \
      coinor-cbc \
    && rm -rf /var/lib/apt/lists/*

# Alias python/pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app
COPY ./code /app/code

# Python deps
# - cupy-cuda12x: prebuilt for CUDA 12.x (compatible with this 12.2 runtime)
RUN python -m pip install --upgrade pip && \
    pip install \
      "brian2tools==0.3" \
      "brian2==2.9.0" \
      "numpy==2.2.6" \
      "tqdm==4.67.1" \
      "pulp==2.8.0" \
      "cupy-cuda12x"

VOLUME ["/nas"]

ENTRYPOINT ["python", "-u", "code/extract_eee_motifs.py"]