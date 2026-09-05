import cv2
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from app.processing.clahe import image_to_base64


def _composite(ssim: float, psnr: float) -> float:
    return 0.5 * ssim + 0.5 * (min(psnr, 50.0) / 50.0)


def reference_metrics(original: np.ndarray, output: np.ndarray) -> dict:
    """Return SSIM and PSNR for the enhanced output compared with the original."""
    ssim = float(structural_similarity(original, output, data_range=255))
    psnr = float(peak_signal_noise_ratio(original, output, data_range=255))
    return {"ssim": ssim, "psnr": psnr}


def compare_enhancements(
    original: np.ndarray,
    clahe_output: np.ndarray,
    cnn_output: np.ndarray,
) -> dict:
    results = {}
    for name, output in [("clahe", clahe_output), ("cnn", cnn_output)]:
        m = reference_metrics(original, output)
        comp = _composite(m["ssim"], m["psnr"])
        results[name] = {**m, "composite": comp}

    winner = "cnn" if results["cnn"]["composite"] >= results["clahe"]["composite"] else "clahe"
    winning_img = cnn_output if winner == "cnn" else clahe_output

    return {
        "clahe_metrics": results["clahe"],
        "cnn_metrics": results["cnn"],
        "winner": winner,
        "winning_image_b64": image_to_base64(winning_img),
        "composite_scores": {
            "clahe": results["clahe"]["composite"],
            "cnn": results["cnn"]["composite"],
        },
    }
