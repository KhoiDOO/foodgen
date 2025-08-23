# Base image (you can change this to another CUDA image if needed)
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

# Set noninteractive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Create Download directory
WORKDIR /Download

# Update & install required packages
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
        nano \
        curl \
        git \
        tmux \
        htop \
        nvtop \
        zlib1g \
        build-essential \
        software-properties-common \
        python3.10-dev && \
    rm -rf /var/lib/apt/lists/*

# Default command (can be overridden at runtime)
CMD ["bash"]

# Clone the repository using the provided token
ARG token
RUN git clone https://${token}@github.com/KhoiDOO/foodgen.git /foodgen

# Set the working directory
WORKDIR /foodgen

