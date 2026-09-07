import logging
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.auth import get_current_user
from app.database import ScanSession, User, get_db
from app.storage import ObjectStorageError, storage
from app.models.enhancer import enhancer
from app.processing.clahe import enhance_clahe, image_to_base64
from app.processing.comparison import reference_metrics
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
async def enhance_clahe_endpoint(file: UploadFile = File(...), session_id: int | None = Query(None), user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    contents = await file.read()
    image = _decode_upload(file, contents)
    start = time.perf_counter()

    before_metrics = compute_metrics(image)
    enhanced = enhance_clahe(image)
    after_metrics = compute_metrics(enhanced)
    ref = reference_metrics(image, enhanced)
    elapsed_ms = (time.perf_counter() - start) * 1000

    response = EnhancementResponse(
        enhanced_image_b64=image_to_base64(enhanced),
        method="clahe",
        before_metrics=image_quality_to_metrics_dict(before_metrics),
        after_metrics=image_quality_to_metrics_dict(
            after_metrics, ssim=ref["ssim"], psnr=ref["psnr"]
        ),
        processing_time_ms=round(elapsed_ms, 2),
    )
    session = db.get(ScanSession, session_id) if session_id else None
    if session and session.user_id != user.id:
        raise HTTPException(status_code=403, detail="You do not own this session")
    if session is None:
        session = ScanSession(user_id=user.id)
        db.add(session)
    db.commit()
    db.refresh(session)

    _, enhanced_bytes = cv2.imencode(".png", enhanced)
    try:
        original_path = session.image_path or storage.upload_image(contents, user.id, session.id, image_type="original")
        enhanced_path = storage.upload_image(enhanced_bytes.tobytes(), user.id, session.id, image_type="enhanced")
    except ObjectStorageError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail="Image storage is temporarily unavailable. Please try again.") from exc
    session.image_path = original_path
    session.enhanced_image_path = enhanced_path
    db.commit()

    response.session_id = session.id
    if original_path:
        response.original_image_url = storage.get_signed_url(original_path)
    if enhanced_path:
        response.enhanced_image_url = storage.get_signed_url(enhanced_path)
    
    return response


@router.post("/enhance/cnn", response_model=EnhancementResponse)
async def enhance_cnn_endpoint(file: UploadFile = File(...), session_id: int | None = Query(None), user: User = Depends(get_current_user), db: Session = Depends(get_db)):
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
    ref = reference_metrics(image, enhanced)
    elapsed_ms = (time.perf_counter() - start) * 1000

    response = EnhancementResponse(
        enhanced_image_b64=image_to_base64(enhanced),
        method="cnn",
        before_metrics=image_quality_to_metrics_dict(before_metrics),
        after_metrics=image_quality_to_metrics_dict(
            after_metrics, ssim=ref["ssim"], psnr=ref["psnr"]
        ),
        processing_time_ms=round(elapsed_ms, 2),
    )
    session = db.get(ScanSession, session_id) if session_id else None
    if session and session.user_id != user.id:
        raise HTTPException(status_code=403, detail="You do not own this session")
    if session is None:
        session = ScanSession(user_id=user.id)
        db.add(session)
    db.commit()
    db.refresh(session)

    _, enhanced_bytes = cv2.imencode(".png", enhanced)
    try:
        original_path = session.image_path or storage.upload_image(contents, user.id, session.id, image_type="original")
        enhanced_path = storage.upload_image(enhanced_bytes.tobytes(), user.id, session.id, image_type="enhanced")
    except ObjectStorageError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail="Image storage is temporarily unavailable. Please try again.") from exc
    session.image_path = original_path
    session.enhanced_image_path = enhanced_path
    db.commit()

    response.session_id = session.id
    if original_path:
        response.original_image_url = storage.get_signed_url(original_path)
    if enhanced_path:
        response.enhanced_image_url = storage.get_signed_url(enhanced_path)
    
    return response
