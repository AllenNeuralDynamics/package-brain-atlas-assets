# hash:sha256:1e6b3c65ff83da094961d08d979c615691ba59c1957c69da2a1dc230121c0cc4
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:code-server4.129.0python3.12.4codeocean-mcp-serveruvpyreflyetc

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    aind-data-schema \
    awscli \
    brainglobe-atlasapi==2.3.0 \
    cloud-volume \
    ngff-zarr \
    nibabel \
    ome-zarr \
    pandas \
    pyarrow \
    simpleitk \
    tensorstore \
    zmesh
