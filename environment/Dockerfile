# hash:sha256:7e10c68bb4dd80b1c9a85b26a440463ea9bebf2f2d9ba49aae5088612ef01b30
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:code-server4.101.2python3.12.4aind-metadata-mcp0.4.2

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.1.0 \
    awscli==1.42.61 \
    cloud-volume==12.7.1 \
    nibabel==5.3.2 \
    ome-zarr==0.12.2 \
    pandas==2.3.3 \
    pyarrow==22.0.0 \
    simpleitk==2.5.2 \
    tensorstore==0.1.78 \
    zmesh==1.9.0
