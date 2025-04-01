FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y python3 python3-pip
WORKDIR /
RUN pip install --no-cache-dir runpod fastapi uvicorn transformers torch huggingface_hub tensorflow tf-keras

COPY rp_handler.py /

# Start the container
CMD ["python3", "-u", "rp_handler.py"]