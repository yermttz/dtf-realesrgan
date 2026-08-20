FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PORT=8000
ENV TMP_DIR=/tmp
ENV WEIGHTS_DIR=/app/weights

# Instalar dependencias del sistema necesarias para OpenCV y procesamiento de imágenes
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar librerías de Python para Real-ESRGAN y el servidor HTTP
RUN pip3 install --no-cache-dir \
    runpod \
    requests \
    opencv-python \
    pillow \
    torchvision \
    basicsr \
    realesrgan \
    fastapi \
    uvicorn \
    python-multipart

COPY handler.py http_app.py config.py validation.py processor.py callback.py pipeline.py public_errors.py logging_utils.py /app/
RUN mkdir -p /app/weights /tmp

CMD ["python3", "-u", "/app/handler.py"]
