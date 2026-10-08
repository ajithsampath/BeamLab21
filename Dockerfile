# ── BeamLab21 ──────────────────────────────────────────────────────────────
# Multi-stage build:
#   base   – runtime deps only (slim image for deployment / CI)
#   dev    – adds pytest + ruff for running tests inside the container
# ───────────────────────────────────────────────────────────────────────────

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

# Keeps Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System deps needed by scipy / matplotlib / h5py
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libhdf5-dev \
        && rm -rf /var/lib/apt/lists/*

# Install Python deps before copying the source so this layer is cached
COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

# Configs and data directories expected by the CLI
COPY configs/ ./configs/

# Outputs directory (write target at runtime)
RUN mkdir -p outputs data

# Default entrypoint: the beamlab21 CLI
ENTRYPOINT ["beamlab21"]
CMD ["--help"]


# ── dev stage ───────────────────────────────────────────────────────────────
FROM base AS dev

RUN pip install --no-cache-dir ".[dev]"

COPY tests/ ./tests/

CMD ["pytest", "-q"]
