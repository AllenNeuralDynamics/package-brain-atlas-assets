# hash:sha256:ac3c056cb8bdcf4b3728c57f61c033df29dab12ee44b87c686dd25abf88c3d4d
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:code-server4.101.2python3.12.4aind-metadata-mcp0.4.2

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.0.1 \
    awscli==1.42.32 \
    cloud-volume==12.5.0 \
    nibabel==5.3.2 \
    ome-zarr==0.12.2 \
    pandas==2.3.2 \
    pyarrow==21.0.0 \
    simpleitk==2.5.2 \
    tensorstore==0.1.76
