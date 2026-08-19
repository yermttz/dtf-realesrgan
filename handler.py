import os
import uuid
import subprocess
import requests
import runpod


MODEL_DIR = "/app/models"
EXECUTABLE = "/app/realesrgan-ncnn-vulkan"


MODELS = {
    "anime": "realesrgan-x4plus-anime",
    "normal": "realesrgan-x4plus"
}


def download_image(url, path):
    response = requests.get(
        url,
        timeout=120
    )

    response.raise_for_status()

    with open(path, "wb") as f:
        f.write(response.content)


def handler(job):

    data = job["input"]

    image_url = data.get("image_url")

    if not image_url:
        return {
            "error": "image_url is required"
        }

    model_type = data.get("model", "anime")
    scale = int(data.get("scale", 4))
    tile = int(data.get("tile", 128))

    if model_type not in MODELS:
        return {
            "error": "Invalid model. Use 'anime' or 'normal'."
        }

    if scale not in [2, 3, 4]:
        return {
            "error": "Invalid scale. Use 2, 3 or 4."
        }

    if tile < 32:
        return {
            "error": "Tile must be >= 32."
        }

    model_name = MODELS[model_type]

    job_id = str(uuid.uuid4())

    input_path = f"/tmp/{job_id}-input"
    output_path = f"/tmp/{job_id}-output.png"

    try:

        # --------------------------------
        # Download input image
        # --------------------------------

        download_image(
            image_url,
            input_path
        )


        # --------------------------------
        # Build Real-ESRGAN command
        # --------------------------------

        command = [
            EXECUTABLE,

            "-i",
            input_path,

            "-o",
            output_path,

            "-n",
            model_name,

            "-s",
            str(scale),

            "-t",
            str(tile),

            "-f",
            "png",

            "-m",
            MODEL_DIR,

            "-g",
            "0"
        ]


        # --------------------------------
        # Run Real-ESRGAN
        # --------------------------------

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=900
        )


        if result.returncode != 0:

            return {
                "error": "Real-ESRGAN failed",
                "return_code": result.returncode,
                "log": result.stdout
            }


        # --------------------------------
        # Verify output
        # --------------------------------

        if not os.path.exists(output_path):

            return {
                "error": "Output image was not created",
                "log": result.stdout
            }


        # --------------------------------
        # Return result
        # --------------------------------

        return {

            "success": True,

            "model": model_type,

            "model_name": model_name,

            "scale": scale,

            "tile": tile,

            "output_file": output_path,

            "log": result.stdout

        }


    except subprocess.TimeoutExpired:

        return {
            "error": "Real-ESRGAN timeout"
        }


    except Exception as e:

        return {
            "error": str(e)
        }


    finally:

        # Remove input file.
        # Output is intentionally kept for the
        # next step where we will upload it.

        if os.path.exists(input_path):

            os.remove(input_path)


runpod.serverless.start({
    "handler": handler
})
