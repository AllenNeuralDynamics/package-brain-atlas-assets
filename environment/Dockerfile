# hash:sha256:e887157fe7033102f93a8776729be0d9e4add5e41b9f69b999d90300cd95b08c
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
    tensorstore==0.1.76 \
    zmesh==1.8.0
