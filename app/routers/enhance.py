import logging
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.models.enhancer import enhancer
from app.processing.clahe import enhance_clahe, image_to_base64
from app.processing.metrics import compute_metrics, image_quality_to_metrics_dict
from app.schemas.responses import EnhancementResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _decode_upload(file: UploadFile, contents: bytes) -> np.ndarray:
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
            detail=f"File too large. Max {settings.MAX_IMAGE_SIZE_MB} MB.",
        )
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise HTTPException(status_code=422, detail="Could not decode image")
    return image


@router.post("/enhance/clahe", response_model=EnhancementResponse)
async def enhance_clahe_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    image = _decode_upload(file, contents)
    start = time.perf_counter()

    before_metrics = compute_metrics(image)
    enhanced = enhance_clahe(image)
    after_metrics = compute_metrics(enhanced)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return EnhancementResponse(
        enhanced_image_b64=image_to_base64(enhanced),
        method="clahe",
        before_metrics=image_quality_to_metrics_dict(before_metrics),
        after_metrics=image_quality_to_metrics_dict(after_metrics),
        processing_time_ms=round(elapsed_ms, 2),
    )


@router.post("/enhance/cnn", response_model=EnhancementResponse)
async def enhance_cnn_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    image = _decode_upload(file, contents)
    start = time.perf_counter()

    before_metrics = compute_metrics(image)
    try:
        enhanced = enhancer.enhance(image)
    except HTTPException as exc:
        if exc.status_code == 503:
            raise HTTPException(
                status_code=503,
                detail="CNN enhancer unavailable. Use /enhance/clahe instead.",
            )
        raise
    after_metrics = compute_metrics(enhanced)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return EnhancementResponse(
        enhanced_image_b64=image_to_base64(enhanced),
        method="cnn",
        before_metrics=image_quality_to_metrics_dict(before_metrics),
        after_metrics=image_quality_to_metrics_dict(after_metrics),
        processing_time_ms=round(elapsed_ms, 2),
    )
