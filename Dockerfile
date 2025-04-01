FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y python3 python3-pip wget
WORKDIR /

# Install CUDA toolkit for libdevice
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-repo-ubuntu2004_11.8.0-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2004_11.8.0-1_amd64.deb && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    apt-get update && apt-get install -y cuda-toolkit-11-8

RUN pip install --no-cache-dir runpod fastapi uvicorn transformers torch huggingface_hub tensorflow tf-keras

COPY rp_handler.py /

# Start the container
CMD ["python3", "-u", "rp_handler.py"]