FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema necesarias para OpenCV y procesamiento de imágenes
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar librerías de Python para Real-ESRGAN y RunPod
RUN pip3 install --no-cache-dir \
    runpod \
    requests \
    opencv-python \
    pillow \
    torchvision \
    basicsr \
    realesrgan

# Copiar el handler de Python
COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
