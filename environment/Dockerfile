# hash:sha256:770fdbc24d992cf9a282c41aa0039a9c5f2cba081139916e1ac50ab4d72465ab
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:code-server4.108.2python3.12.4aind-metadata-mcp0.5.2withDOTFILES_REPOsupport

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.6.0 \
    awscli==1.44.58 \
    brainglobe-atlasapi==2.3.0 \
    cloud-volume==12.12.0 \
    nibabel==5.4.2 \
    ome-zarr==0.14.0 \
    pandas==3.0.1 \
    pyarrow==23.0.1 \
    simpleitk==2.5.3 \
    tensorstore==0.1.82 \
    zmesh==1.11.0
