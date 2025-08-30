# hash:sha256:3e0a5d652c0d0d5e0f08355049c87908609888bd385964f99d334266986f4b5e
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:code-server4.101.2python3.12.4aind-metadata-mcp0.4.2

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema \
    awscli \
    cloud-volume \
    nibabel \
    ome-zarr \
    pandas \
    pyarrow \
    simpleitk \
    tensorstore
