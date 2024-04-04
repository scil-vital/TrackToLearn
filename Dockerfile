FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git build-essential \
      liblapack-dev libopenblas-dev libgl1 libxrender1

RUN git clone https://github.com/scil-vital/TrackToLearn.git

WORKDIR /TrackToLearn

RUN pip install Cython numpy packaging
RUN pip install -e .
