import os
import uuid
import subprocess
import requests
import runpod


MODEL_DIR = "/app/models"
EXECUTABLE = "/app/realesrgan-ncnn-vulkan"


def download_image(url, path):
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    with open(path, "wb") as f:
        f.write(response.content)


def handler(job):
    data = job["input"]

    image_url = data["image_url"]
    model = data.get("model", "normal")

    if model == "anime":
        model_name = "realesrgan-x4plus-anime"
    elif model == "normal":
        model_name = "realesrgan-x4plus"
    else:
        return {
            "success": False,
            "error": "Invalid model. Use 'anime' or 'normal'."
        }

    job_id = str(uuid.uuid4())

    input_path = f"/tmp/{job_id}-input"
    output_path = f"/tmp/{job_id}-output.png"

    try:

        gpu_info = subprocess.run(
            ["vulkaninfo", "--summary"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30
        )

        print("=== VULKAN INFO ===")
        print(gpu_info.stdout)
        print("===================")

        download_image(image_url, input_path)

        command = [
            EXECUTABLE,
            "-i", input_path,
            "-o", output_path,
            "-n", model_name,
            "-s", "4",
            "-t", "128",
            "-f", "png",
            "-m", MODEL_DIR,
            "-g", "0"
        ]

        print("Running command:")
        print(" ".join(command))

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600
        )

        print(result.stdout)

        if result.returncode != 0:
            return {
                "success": False,
                "error": "Real-ESRGAN failed",
                "returncode": result.returncode,
                "log": result.stdout
            }

        if not os.path.exists(output_path):
            return {
                "success": False,
                "error": "Output image was not created",
                "log": result.stdout
            }

        return {
            "success": True,
            "output_file": output_path,
            "model": model,
            "model_name": model_name,
            "scale": 4,
            "tile": 128,
            "log": result.stdout
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


runpod.serverless.start({
    "handler": handler
})
