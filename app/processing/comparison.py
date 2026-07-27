import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from app.processing.clahe import image_to_base64

def _brisque_score(image: np.ndarray) -> float:
    try:
        from brisque import BRISQUE
        return float(BRISQUE().score(image))
    except Exception:
        return -1.0
def _composite(ssim: float, psnr: float, brisque: float) -> float:
    if brisque < 0:
        return 0.4 * ssim + 0.4 * (min(psnr, 50.0) / 50.0)
    return (
        0.4 * ssim
        + 0.4 * (min(psnr, 50.0) / 50.0)
        + 0.2 * (1.0 - max(0.0, min(brisque, 100.0)) / 100.0)
    )
def compare_enhancements(
    original: np.ndarray,
    clahe_output: np.ndarray,
    cnn_output: np.ndarray,
) -> dict:
    results = {}
    for name, output in [("clahe", clahe_output), ("cnn", cnn_output)]:
        ssim = float(structural_similarity(original, output, data_range=255))
        psnr = float(peak_signal_noise_ratio(original, output, data_range=255))
        brisque = _brisque_score(output)
        comp = _composite(ssim, psnr, brisque)
        results[name] = {
            "ssim": ssim,
            "psnr": psnr,
            "brisque": brisque,
            "composite": comp,
        }

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
