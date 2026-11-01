# syntax=docker/dockerfile:1
# seaice-predict on CPU: AI4Arctic scene (NetCDF) -> georeferenced COGs.
#
#   docker run --rm -v "$PWD:/data" ghcr.io/reddytian/sea-ice-segmentation SCENE.nc --out out/
#
# Layers go from least to most frequently changed: PyTorch, model, code.
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# CPU-only PyTorch: the default Linux wheel would add ~2.5 GB of CUDA libraries.
# 2.7.1 is the version scripts/check_parity.py was run with on all 24 test scenes.
ARG TORCH_VERSION=2.7.1
RUN pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu

# Model bundle from the GitHub release, verified against the checksum in the repo.
COPY docker/fetch_model.py docker/model.json /opt/seaice/
RUN python /opt/seaice/fetch_model.py /opt/seaice/model.pt
ENV SEAICE_MODEL=/opt/seaice/model.pt

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install . && rm -rf /build

RUN useradd --create-home --uid 1000 seaice
USER seaice
WORKDIR /data
ENTRYPOINT ["seaice-predict"]
CMD ["--help"]
