# Use the high-performance PyTorch 2.4 / CUDA 12.4 base (Ubuntu 22.04)
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Bake uv into the container for rapid package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# To avoid interaction with timezone etc.
ARG DEBIAN_FRONTEND=noninteractive

# Install ESSENTIAL system dependencies for medical image processing
# and standard development tools (OpenSSL, Git, etc.)
RUN apt-get update && apt-get install -y \
    libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev \
    libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev \
    libharfbuzz-dev libfribidi-dev libxcb1-dev \
    expect openssl curl git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to the mount point
WORKDIR /workspace

# No CMD is needed; start.sh will invoke 'sleep infinity' to keep it alive as a background worker
