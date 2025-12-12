# hash:sha256:401fece4e6572446c0ea29ac716b0285b48f4a6f31c762b16bf949e69f550384
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:code-server4.104.3python3.12.4aind-metadata-mcp0.5.2withDOTFILES_REPOsupport

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.2.0 \
    awscli==1.43.4 \
    cloud-volume==12.8.0 \
    nibabel==5.3.2 \
    ome-zarr==0.12.2 \
    pandas==2.3.3 \
    pyarrow==22.0.0 \
    simpleitk==2.5.3 \
    tensorstore==0.1.79 \
    zmesh==1.9.0
