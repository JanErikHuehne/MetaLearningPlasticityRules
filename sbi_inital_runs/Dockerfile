# Use Python 3.10 slim image
FROM python:3.10-slim

LABEL org.opencontainers.image.title="Container for Jan - SBI Initial Runs" \
      org.opencontainers.image.description="Run a single SBI script with pinned dependencies" \
      org.opencontainers.image.authors="jan.huehne@tum.de" \
      org.opencontainers.image.licenses="MIT"

# Non-interactive installs and cleaner logs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    nas_path=/nas

# System build tools (needed for packages with native extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# App folder
WORKDIR /app

# Copy only your code (keeps image smaller)
# Assumes Docker build context is the repo root containing ./code
COPY ./code /app/code

# Python dependencies (pinned)
RUN python -m pip install --upgrade pip && \
    pip install \
      "brian2tools==0.3" \
      "brian2==2.9.0" \
      "numpy==2.2.6" \
      "tqdm==4.67.1"

# Optional: declare a volume for /nas (host will mount it)
VOLUME ["/nas"]

# Default run: execute your script
ENTRYPOINT ["python", "-u", "code/run_single.py"]
