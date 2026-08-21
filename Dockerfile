FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TMP_DIR=/tmp
ENV WEIGHTS_DIR=/app/weights
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Wheels oficiales cu128 (incluyen sm_120). --no-deps evita que pip traiga torch CPU de PyPI.
RUN pip3 install --no-cache-dir --no-deps \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.9.1+cu128 \
    torchvision==0.24.1+cu128

# Deps controladas; realesrgan/basicsr con --no-deps para no pisar torch CUDA.
RUN pip3 install --no-cache-dir \
    numpy==1.26.4 \
    pillow==11.1.0 \
    opencv-python-headless==4.10.0.84 \
    requests==2.32.4 \
    tqdm==4.67.1 \
    pyyaml==6.0.2 \
    addict==2.4.0 \
    future==1.0.0 \
    scipy==1.15.3 \
    runpod==1.7.13 \
 && pip3 install --no-cache-dir --no-deps \
    basicsr==1.4.2 \
    realesrgan==0.3.0

# Fallar el build si pip dejó un torch CPU. sm_120 se valida en runtime con GPU (check_cuda.py).
RUN python3 -c "import torch, torchvision; v=torch.__version__; assert '+cu128' in v and 'cpu' not in v, v; assert torchvision.__version__.startswith('0.24.1'), torchvision.__version__"

COPY handler.py config.py validation.py processor.py callback.py pipeline.py downloader.py public_errors.py logging_utils.py torchvision_compat.py cuda_compat.py model_weights.py check_cuda.py check_realesrgan.py /app/
COPY weights/ /app/weights/
RUN test -f /app/weights/RealESRGAN_x4plus.pth \
 && test -f /app/weights/RealESRGAN_x4plus_anime_6B.pth
RUN mkdir -p /tmp

CMD ["python3", "-u", "/app/handler.py"]
