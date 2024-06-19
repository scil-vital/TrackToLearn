FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git build-essential \
      liblapack-dev libopenblas-dev libgl1 libxrender1

RUN git clone https://github.com/scil-vital/TrackToLearn.git

WORKDIR /TrackToLearn

RUN pip install Cython==0.29.* numpy==1.25.* packaging --quiet
RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --extra-index-url https://download.pytorch.org/whl/cu118 --quiet
RUN pip install -e .
