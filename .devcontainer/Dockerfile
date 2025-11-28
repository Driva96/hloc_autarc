FROM colmap/colmap:latest
MAINTAINER Andre Dueding
ARG PYTHON_VERSION=3.11
RUN apt-get update -y
RUN apt-get install -y \
        unzip \
        wget \
        software-properties-common\
        git &&\
    update-ca-certificates
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y update && \
    apt-get install -y python${PYTHON_VERSION}

# Configure git for better SSL handling
RUN git config --global http.sslVerify true && \
    git config --global http.postBuffer 524288000


RUN wget https://bootstrap.pypa.io/get-pip.py && python${PYTHON_VERSION} get-pip.py
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*
COPY . /app
WORKDIR /app
RUN python -m pip install --upgrade pip
RUN python -m pip install pycolmap
RUN python -m pip install -r requirements.txt
RUN python -m pip install notebook