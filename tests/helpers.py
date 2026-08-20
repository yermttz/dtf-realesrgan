import io
from dataclasses import dataclass

from PIL import Image


JOB_ID = "25a68587-1ab3-404e-8b05-f3926bacb25b"
IMAGE_ID = "1787248816746-hgiat1"
USER_ID = "71872588"
FILE_NAME = "Spider Man Brand New Day__1787248816746-hgiat1.jpg"
AUTH = "test-worker-token"
CALLBACK_SECRET = "result-secret"


def jpeg_bytes(width: int = 524, height: int = 698) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (40, 80, 160)).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def png_bytes(width: int = 32, height: int = 32) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", (width, height), (9, 9, 9, 255)).save(buf, format="PNG")
    return buf.getvalue()


def kanonico_fields(**overrides):
    fields = {
        "job_id": JOB_ID,
        "image_id": IMAGE_ID,
        "user_id": USER_ID,
        "model": "anime",
        "scale": "4",
        "file_name": FILE_NAME,
        "image_mime": "image/jpeg",
    }
    fields.update(overrides)
    return fields


def kanonico_multipart(image: bytes | None = None, **overrides):
    data = kanonico_fields(**overrides)
    files = None
    if image is not None:
        files = {"file": (FILE_NAME, image, "image/jpeg")}
    return data, files


@dataclass
class FakeProcessor:
    fail: bool = False
    seen_paths: list | None = None

    def enhance(self, input_path: str, output_path: str, model: str, scale: int):
        if self.seen_paths is not None:
            self.seen_paths.append((input_path, output_path))
        if self.fail:
            raise RuntimeError("CUDA OOM in Real-ESRGAN on RunPod GPU worker endpoint")
        with Image.open(input_path) as img:
            w, h = img.size
            out = img.convert("RGBA").resize((w * scale, h * scale), Image.NEAREST)
            out.save(output_path, "PNG")
        return w, h, w * scale, h * scale
