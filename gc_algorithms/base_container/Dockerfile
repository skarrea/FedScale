# syntax=docker/dockerfile:1

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ARG MODEL=umamba_mtl

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt /tmp/requirements-runtime.txt

RUN python -m pip install --no-cache-dir --upgrade \
        "pip<25" \
        "setuptools<81" \
        "wheel<0.46" \
        packaging

RUN python -m pip install --no-cache-dir -r /tmp/requirements-runtime.txt

RUN python -m pip install --no-cache-dir --no-deps \
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.2.0.post2/causal_conv1d-1.2.0.post2+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
    "https://github.com/state-spaces/mamba/releases/download/v1.2.0.post1/mamba_ssm-1.2.0.post1+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

COPY packages/umamba /opt/umamba
RUN python -m pip install --no-cache-dir /opt/umamba

WORKDIR /opt/ml/model
COPY models/${MODEL}/ /opt/ml/model/
COPY entrypoint.sh /opt/ml/model/entrypoint.sh
COPY mamba_smoke_test.py /opt/ml/model/mamba_smoke_test.py

RUN chmod +x /opt/ml/model/entrypoint.sh

ENTRYPOINT ["/opt/ml/model/entrypoint.sh"]
