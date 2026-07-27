import numpy as np
import cv2
from scipy import stats
from skimage.restoration import estimate_sigma

from app.schemas.responses import MetricsDict

def compute_metrics(image: np.ndarray) -> dict:
    """Compute IQA metrics. Accepts float [0,1] or uint8 [0,255] grayscale."""
    if image.dtype == np.uint8:
        float_img = image.astype(np.float32) / 255.0
    else:
        float_img = image.astype(np.float32)
        if float_img.max() > 1.0:
            float_img = float_img / 255.0

    uint8_img = (float_img * 255).astype(np.uint8)

    laplacian_variance = float(cv2.Laplacian(uint8_img, cv2.CV_64F).var())
    rms_contrast = float(np.sqrt(np.mean((float_img - float_img.mean()) ** 2)) * 255)

    hist, _ = np.histogram(uint8_img, bins=256, range=(0, 256))
    hist_norm = hist / (hist.sum() + 1e-10)
    shannon_entropy = float(stats.entropy(hist_norm + 1e-10))

    p5, p95 = np.percentile(uint8_img, [5, 95])
    histogram_spread = float(p95 - p5)

    noise_variance = float(estimate_sigma(float_img, average_sigmas=True))

    if laplacian_variance < 50 or noise_variance > 0.05 or rms_contrast < 20:
        quality_class = "Poor"
    elif laplacian_variance > 200 and rms_contrast > 60 and noise_variance < 0.01:
        quality_class = "Good"
    else:
        quality_class = "Fair"

    defects: list[str] = []
    if laplacian_variance < 50:
        defects.append("motion_blur")
    if noise_variance > 0.05:
        defects.append("excessive_noise")
    if rms_contrast < 20:
        defects.append("underexposure")
    if histogram_spread < 80:
        defects.append("low_contrast")
    if histogram_spread > 230:
        defects.append("overexposure")

    return {
        "laplacian_variance": laplacian_variance,
        "rms_contrast": rms_contrast,
        "shannon_entropy": shannon_entropy,
        "histogram_spread": histogram_spread,
        "noise_variance": noise_variance,
        "quality_class": quality_class,
        "defects": defects,
    }


def image_quality_to_metrics_dict(metrics: dict) -> MetricsDict:
    return MetricsDict(
        laplacian_variance=metrics["laplacian_variance"],
        rms_contrast=metrics["rms_contrast"],
        shannon_entropy=metrics["shannon_entropy"],
        histogram_spread=metrics["histogram_spread"],
        noise_variance=metrics["noise_variance"],
    )
