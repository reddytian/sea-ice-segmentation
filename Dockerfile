# syntax=docker/dockerfile:1
# seaice-predict on CPU: AI4Arctic scene (NetCDF) -> georeferenced COGs.
#
#   docker run --rm -v "$PWD:/data" ghcr.io/reddytian/sea-ice-segmentation SCENE.nc --out out/
#
# Targets: `cli` (default, the published image) and `lambda` (AWS Lambda, see deploy/aws/).
# Both share one `base`, so the Lambda function runs exactly the code and model of the CLI.
# Layers go from least to most frequently changed: PyTorch, model, code.
FROM python:3.11-slim AS base

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# rasterio's wheel bundles GDAL but links the system libexpat, which the slim image
# lacks (caught by the CI smoke test: "libexpat.so.1: cannot open shared object file").
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

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

# ---- AWS Lambda: S3 event -> seaice.aws_lambda.handler (only /tmp is writable there)
FROM base AS lambda
RUN pip install "awslambdaric==4.2.3" "boto3==1.43.123"
ENV HOME=/tmp
WORKDIR /tmp
ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["seaice.aws_lambda.handler"]

# ---- command-line image (default target, published to GHCR)
FROM base AS cli
RUN useradd --create-home --uid 1000 seaice
USER seaice
WORKDIR /data
ENTRYPOINT ["seaice-predict"]
CMD ["--help"]
