# hash:sha256:3131441ab2b355bf2b166dc254d500df97faf86763bdea54199cd00ec7152606
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:code-server4.101.2python3.12.4aind-metadata-mcp0.4.2

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
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
