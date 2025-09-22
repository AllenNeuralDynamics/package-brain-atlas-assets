# hash:sha256:37282d8e8747a7e2c5fc1e0d7ebe424030fa8d31b26f89993b98c97395596a00
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:code-server4.101.2python3.12.4aind-metadata-mcp0.4.2

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.0.1 \
    awscli==1.42.21 \
    cloud-volume==12.4.1 \
    nibabel==5.3.2 \
    ome-zarr==0.12.2 \
    pandas==2.3.2 \
    pyarrow==21.0.0 \
    simpleitk==2.5.2 \
    tensorstore==0.1.76
