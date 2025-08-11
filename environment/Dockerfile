# hash:sha256:0d5c06a1c5ff9f1baced44b70586b40c9abb102815160769c72bb2030dd9026b
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:code-server4.101.2python3.12.4aind-metadata-mcp0.4.2

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.0.0 \
    awscli==1.42.6 \
    cloud-volume==12.4.1 \
    nibabel==5.3.2 \
    ome-zarr==0.11.1 \
    pandas==2.3.1 \
    pyarrow==21.0.0 \
    simpleitk==2.5.2 \
    tensorstore==0.1.76
