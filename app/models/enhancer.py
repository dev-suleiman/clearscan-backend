import logging
from pathlib import Path
import numpy as np
from fastapi import HTTPException
from app.config import settings
logger = logging.getLogger(__name__)
class ImageEnhancer:
    def __init__(self):
        self.model = None
        model_path = Path(settings.MODEL_DIR) / "enhancement_autoencoder.h5"
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(str(model_path), compile=False)
            logger.info("Enhancement autoencoder loaded from %s", model_path)
        except Exception as exc:
            logger.warning("Enhancement autoencoder not found — CNN enhancement unavailable (%s)", exc)
    def enhance(self, image: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise HTTPException(
                status_code=503,
                detail="CNN enhancer unavailable — use CLAHE mode",)
        import cv2
        resized = cv2.resize(image, (512, 512)).astype(np.float32) / 255.0
        batch = resized[np.newaxis, ..., np.newaxis]
        output = self.model.predict(batch, verbose=0)[0, ..., 0]
        result = np.clip(output * 255, 0, 255).astype(np.uint8)
        if result.shape != image.shape:
            result = cv2.resize(result, (image.shape[1], image.shape[0]))
        return result
enhancer = ImageEnhancer()
