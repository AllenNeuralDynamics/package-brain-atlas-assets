# hash:sha256:986f4490ad787777d5f6eddd6a7474e42fdf5cd8c1e4afb13f1bd72b67dfdf2d
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:code-server4.129.0python3.12.4codeocean-mcp-serveruvpyreflyetc

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.9.0 \
    awscli==1.46.0 \
    brainglobe-atlasapi==2.3.0 \
    cloud-volume==12.14.4 \
    ngff-zarr==0.42.1 \
    nibabel==5.4.2 \
    ome-zarr==0.18.0 \
    pandas==3.0.5 \
    pyarrow==25.0.1 \
    simpleitk==2.5.6 \
    tensorstore==0.1.85 \
    zmesh==1.15.0
