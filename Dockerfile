FROM python:3.10-slim

WORKDIR /
RUN pip install --no-cache-dir runpod fastapi uvicorn transformers torch huggingface_hub tensorflow tf-keras

COPY rp_handler.py /

# Start the container
CMD ["python3", "-u", "rp_handler.py"]