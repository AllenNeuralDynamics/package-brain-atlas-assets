# hash:sha256:de5e0d3bb725cd29497d2e0dc91d1e185962a32d4806eb252c5041f59fa59df5
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
    igneous-pipeline==4.36.2 \
    ngff-zarr==0.42.1 \
    nibabel==5.4.2 \
    pandas==3.0.5 \
    pyarrow==25.0.1 \
    simpleitk==2.5.6 \
    tensorstore==0.1.85 \
    zmesh==1.15.0
