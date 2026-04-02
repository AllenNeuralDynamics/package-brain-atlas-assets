# hash:sha256:3cd4db1c4997cea3ea6925b6be6e7993370e12ddf28da120c704c3c7e752a2ad
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:v5.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.6.0 \
    awscli==1.44.71 \
    brainglobe-atlasapi==2.3.0 \
    cloud-volume==12.12.1 \
    nibabel==5.4.2 \
    ome-zarr==0.14.0 \
    pandas==3.0.2 \
    pyarrow==23.0.1 \
    simpleitk==2.5.3 \
    tensorstore==0.1.82 \
    zmesh==1.11.0
