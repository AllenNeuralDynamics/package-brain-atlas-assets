# hash:sha256:e9346f18369561035bfeec6e14762af614b4c3aa0360bf75f2960c7a0533fa63
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:code-server4.108.2python3.12.4aind-metadata-mcp0.5.2withDOTFILES_REPOsupport

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.4.0 \
    awscli==1.44.32 \
    cloud-volume==12.9.2 \
    nibabel==5.3.3 \
    ome-zarr==0.12.2 \
    pandas==3.0.0 \
    pyarrow==23.0.0 \
    simpleitk==2.5.3 \
    tensorstore==0.1.80 \
    zmesh==1.9.0
