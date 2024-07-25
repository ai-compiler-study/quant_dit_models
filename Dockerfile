FROM nvcr.io/nvidia/pytorch:24.06-py3
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y libgl1 libglib2.0-0 wget git git-lfs python3-pip python-is-python3 libcairo2-dev pkg-config python3-dev tzdata && \
    rm -rf /var/lib/apt/lists/*

RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t robbyrussell

# Install dependencies
RUN apt-get install -y build-essential cmake git pkg-config libjpeg-dev \
    libtiff5-dev libpng-dev libavcodec-dev libavformat-dev \
    libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libfontconfig1-dev \
    libcairo2-dev libgdk-pixbuf2.0-dev libpango1.0-dev libgtk2.0-dev libgtk-3-dev \
    libatlas-base-dev gfortran ffmpeg libmagickwand-dev

RUN pip install -U opencv-python-headless diffusers transformers onnx_graphsurgeon gpustat \
    accelerate ipdb tqdm fire ipython jupyter requests numpy matplotlib seaborn

RUN pip install git+https://github.com/huggingface/diffusers
RUN pip install -U sentencepiece protobuf numpy

WORKDIR /app

