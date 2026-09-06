import logging
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.auth import get_current_user
from app.database import ScanSession, User, get_db
from app.storage import storage
from app.models.enhancer import enhancer
from app.processing.clahe import enhance_clahe, image_to_base64
from app.processing.comparison import compare_enhancements
from app.processing.metrics import compute_metrics
from app.schemas.responses import ComparisonResponse

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

@router.post("/compare", response_model=ComparisonResponse)
async def compare(file: UploadFile = File(...), user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    contents = await file.read()
    image = _decode_upload(file, contents)
    start = time.perf_counter()

    clahe_output = enhance_clahe(image)
    fallback_reason: str | None = None
    try:
        cnn_output = enhancer.enhance(image)
    except HTTPException as exc:
        if exc.status_code == 503:
            fallback_reason = "CNN model unavailable"
            cnn_output = clahe_output
        else:
            raise

    result = compare_enhancements(image, clahe_output, cnn_output)
    elapsed_ms = (time.perf_counter() - start) * 1000

    winner = "clahe" if fallback_reason else result["winner"]

    clahe_b64 = image_to_base64(clahe_output)
    cnn_b64 = image_to_base64(cnn_output) if fallback_reason is None else clahe_b64

    response = ComparisonResponse(
        winner=winner,
        winning_image_b64=result["winning_image_b64"] if not fallback_reason else clahe_b64,
        clahe_result={
            **result["clahe_metrics"],
            "image_b64": clahe_b64,
        },
        cnn_result={
            **result["cnn_metrics"],
            "image_b64": cnn_b64,
        },
        composite_scores=result["composite_scores"],
        processing_time_ms=round(elapsed_ms, 2),
        fallback_reason=fallback_reason,
    )
    session = ScanSession(user_id=user.id, enhancement_method=winner, mode="compare")
    db.add(session)
    db.commit()
    db.refresh(session)

    _, winning_bytes = cv2.imencode(".png", clahe_output if winner == "clahe" else cnn_output)
    original_path = storage.upload_image(contents, user.id, session.id, image_type="original")
    winning_path = storage.upload_image(winning_bytes.tobytes(), user.id, session.id, image_type="enhanced")
    session.image_path = original_path
    session.enhanced_image_path = winning_path
    db.commit()
    
    return response
