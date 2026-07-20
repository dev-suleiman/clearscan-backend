import logging
from pathlib import Path

import numpy as np
from fastapi import HTTPException

from app.config import settings
from app.processing.metrics import compute_metrics

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Good", "Fair", "Poor"]


def _defects_from_class(quality_class: str, image: np.ndarray) -> list[str]:
    if quality_class == "Good":
        return []
    if quality_class == "Poor":
        m = compute_metrics(image)
        return m["defects"]
    # Fair — return the single worst metric as a hint
    m = compute_metrics(image)
    defects = m["defects"]
    return defects[:1] if defects else []


class QualityClassifier:
    def __init__(self):
        self.model = None
        model_path = Path(settings.MODEL_DIR) / "quality_classifier.h5"
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(str(model_path))
            logger.info("Quality classifier loaded from %s", model_path)
        except Exception as exc:
            logger.warning("Quality classifier not found — CNN mode unavailable (%s)", exc)

    def predict(self, image: np.ndarray) -> dict:
        if self.model is None:
            raise HTTPException(
                status_code=503,
                detail="CNN model unavailable — use offline mode",
            )
        import cv2
        import tensorflow as tf

        resized = cv2.resize(image, (224, 224))
        rgb = np.stack([resized, resized, resized], axis=-1).astype(np.float32) / 255.0
        batch = rgb[np.newaxis]
        probs = self.model.predict(batch, verbose=0)[0]
        idx = int(np.argmax(probs))
        quality_class = CLASS_NAMES[idx]

        return {
            "quality_class": quality_class,
            "confidence": float(probs[idx]),
            "class_probabilities": {
                "Good": float(probs[0]),
                "Fair": float(probs[1]),
                "Poor": float(probs[2]),
            },
            "defects": _defects_from_class(quality_class, image),
        }


classifier = QualityClassifier()
