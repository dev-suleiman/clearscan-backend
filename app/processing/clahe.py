import base64

import cv2
import numpy as np
from skimage.filters import unsharp_mask


def enhance_clahe(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE + denoising + unsharp masking to a grayscale uint8 image."""
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got dtype {image.dtype}")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    denoised = cv2.fastNlMeansDenoising(
        enhanced, h=7, templateWindowSize=7, searchWindowSize=21
    )

    sharp_float = unsharp_mask(
        denoised.astype(np.float32) / 255.0, radius=1.0, amount=1.5
    )
    result = np.clip(sharp_float * 255, 0, 255).astype(np.uint8)
    return result


def image_to_base64(image: np.ndarray) -> str:
    success, buf = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Failed to encode image to PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def base64_to_image(b64: str) -> np.ndarray:
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to decode base64 image")
    return img
