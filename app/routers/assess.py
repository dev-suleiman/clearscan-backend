import logging
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.models.classifier import classifier
from app.processing.metrics import compute_metrics, image_quality_to_metrics_dict
from app.schemas.responses import QualityAssessmentResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _validate_upload(file: UploadFile, contents: bytes) -> None:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )
    max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(contents)} bytes). Max {settings.MAX_IMAGE_SIZE_MB} MB.",
        )


@router.post("/assess", response_model=QualityAssessmentResponse)
async def assess(file: UploadFile = File(...)):
    contents = await file.read()
    _validate_upload(file, contents)

    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise HTTPException(status_code=422, detail="Could not decode image")

    start = time.perf_counter()
    metrics = compute_metrics(image)
    mode = "offline"
    result: dict = {
        "quality_class": metrics["quality_class"],
        "confidence": 1.0,
        "class_probabilities": {"Good": 0.0, "Fair": 0.0, "Poor": 0.0},
        "defects": metrics["defects"],
    }
    result["class_probabilities"][metrics["quality_class"]] = 1.0

    try:
        cnn = classifier.predict(image)
        result = cnn
        mode = "cnn"
    except HTTPException as exc:
        if exc.status_code != 503:
            raise
        logger.info("CNN unavailable, falling back to offline metrics")

    elapsed_ms = (time.perf_counter() - start) * 1000

    response = QualityAssessmentResponse(
        quality_class=result["quality_class"],
        confidence=result["confidence"],
        class_probabilities=result["class_probabilities"],
        defects=result["defects"],
        metrics=image_quality_to_metrics_dict(metrics),
        processing_time_ms=round(elapsed_ms, 2),
        mode=mode,
    )
    logger.info(
        "assess | class=%s mode=%s time=%.1fms",
        response.quality_class,
        mode,
        elapsed_ms,
    )
    return response
