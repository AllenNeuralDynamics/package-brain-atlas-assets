# hash:sha256:445482140b82ab5f98423e07243067eabfb7e13140eb489a9747725c10ba145d
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
    ngff-zarr==0.41.1 \
    pandas==3.0.2 \
    pyarrow==23.0.1 \
    simpleitk==2.5.3 \
    tensorstore==0.1.82 \
    zmesh==1.11.0
