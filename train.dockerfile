# Base image (you can change this to another CUDA image if needed)
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

# Set noninteractive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Update & install required packages
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y nano curl git zlib1g build-essential software-properties-common
RUN apt-get install -y python3-pip
RUN pip install -U pip wheel
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
RUN pip install tqdm beartype 
RUN pip install einops==0.8.1
RUN pip install ema-pytorch==0.7.7
RUN pip install memory-efficient-attention-pytorch==0.1.6
RUN pip install pillow
RUN pip install sentencepiece==0.2.1
RUN pip install transformers==4.55.3
RUN pip install vector-quantize-pytorch==1.23.1
RUN pip install accelerate==1.10.0
RUN pip install omegaconf==2.3.0

WORKDIR /git

RUN git clone https://github.com/KhoiDOO/foodgen.git

WORKDIR /git/foodgen

CMD bash/script/vqganvae.sh