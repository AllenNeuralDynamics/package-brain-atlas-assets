# hash:sha256:4628df4bbb7a99918e6870ae7073d20457d25e6d0cb17bf21df65e6a2607a66d
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:code-server4.101.2python3.12.4aind-metadata-mcp0.4.2

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

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
