import os
import uuid
import torch
import requests
import runpod
from PIL import Image
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_archs import RRDBNet

# Verificar si CUDA está disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"=== USANDO DISPOSITIVO: {device} ===")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def download_image(url, path):
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    with open(path, "wb") as f:
        f.write(response.content)

def handler(job):
    data = job["input"]
    image_url = data["image_url"]
    model = data.get("model", "normal")

    # Configurar modelo según la opción
    if model == "anime":
        # x4plus_anime_6B para anime/ilustraciones
        model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        # Descarga automática o uso de pesos oficiales
        file_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
    else:
        # x4plus estándar para fotos/general
        model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'

    job_id = str(uuid.uuid4())
    input_path = f"/tmp/{job_id}-input.jpg"
    output_path = f"/tmp/{job_id}-output.png"

    try:
        download_image(image_url, input_path)

        # Inicializar el Upsampler con PyTorch
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=file_url,
            model=model_arch,
            tile=128,
            tile_pad=10,
            pre_pad=0,
            half=True if torch.cuda.is_available() else False # Usar FP16 para mayor velocidad en GPU
        )

        # Leer imagen
        img = Image.open(input_path).convert('RGB')
        img_np = np.array(img)

        # Procesar con IA
        output, _ = upsampler.enhance(img_np, outscale=4)

        # Guardar resultado
        result_img = Image.fromarray(output)
        result_img.save(output_path, "PNG")

        if not os.path.exists(output_path):
            return {
                "success": False,
                "error": "Output image was not created"
            }

        return {
            "success": True,
            "output_file": output_path,
            "model": model,
            "scale": 4
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

runpod.serverless.start({
    "handler": handler
})
