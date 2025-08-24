# Base image (you can change this to another CUDA image if needed)
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

# Set noninteractive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Update & install required packages
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y nano curl git tmux zlib1g build-essential software-properties-common
RUN apt-get install -y python3-pip

# CMD ["pip", "--version"]

RUN pip install pillow tqdm

# CD to data
WORKDIR /data

CMD python3 download.py