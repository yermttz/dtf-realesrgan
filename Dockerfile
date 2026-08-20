FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libvulkan1 \
    vulkan-tools \
    mesa-vulkan-drivers \
    libnvidia-gl-550 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY realesrgan-ncnn-vulkan /app/realesrgan-ncnn-vulkan
COPY models /app/models
COPY handler.py /app/handler.py

RUN chmod +x /app/realesrgan-ncnn-vulkan

RUN pip3 install --no-cache-dir \
    runpod \
    requests

CMD ["python3", "/app/handler.py"]
