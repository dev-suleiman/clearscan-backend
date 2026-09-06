"""History endpoint for retrieving user's scan sessions with image URLs."""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import ScanSession, User, get_db
from app.storage import storage
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


class ImageInfo(BaseModel):
    url: str | None = None
    path: str | None = None


class ScanHistoryItem(BaseModel):
    id: int
    quality_class: str | None
    enhancement_method: str | None
    mode: str | None
    created_at: datetime
    original_image: ImageInfo | None = None
    enhanced_image: ImageInfo | None = None


class UserHistoryResponse(BaseModel):
    user_id: int
    user_name: str
    total_scans: int
    history: list[ScanHistoryItem]


@router.get("/history", response_model=UserHistoryResponse)
async def get_user_history(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get all scan sessions for the current user with image URLs."""
    sessions = db.query(ScanSession).filter(ScanSession.user_id == user.id).order_by(ScanSession.created_at.desc()).all()

    history = []
    for session in sessions:
        original_image = None
        enhanced_image = None

        if session.image_path:
            original_url = storage.get_signed_url(session.image_path)
            original_image = ImageInfo(url=original_url, path=session.image_path)

        if session.enhanced_image_path:
            enhanced_url = storage.get_signed_url(session.enhanced_image_path)
            enhanced_image = ImageInfo(url=enhanced_url, path=session.enhanced_image_path)

        history.append(
            ScanHistoryItem(
                id=session.id,
                quality_class=session.quality_class,
                enhancement_method=session.enhancement_method,
                mode=session.mode,
                created_at=session.created_at,
                original_image=original_image,
                enhanced_image=enhanced_image,
            )
        )

    return UserHistoryResponse(
        user_id=user.id,
        user_name=user.name,
        total_scans=len(sessions),
        history=history,
    )


@router.get("/history/{session_id}", response_model=ScanHistoryItem)
async def get_session_detail(session_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get details of a specific scan session (only if it belongs to the user)."""
    session = db.query(ScanSession).filter(ScanSession.id == session_id, ScanSession.user_id == user.id).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    original_image = None
    enhanced_image = None

    if session.image_path:
        original_url = storage.get_signed_url(session.image_path)
        original_image = ImageInfo(url=original_url, path=session.image_path)

    if session.enhanced_image_path:
        enhanced_url = storage.get_signed_url(session.enhanced_image_path)
        enhanced_image = ImageInfo(url=enhanced_url, path=session.enhanced_image_path)

    return ScanHistoryItem(
        id=session.id,
        quality_class=session.quality_class,
        enhancement_method=session.enhancement_method,
        mode=session.mode,
        created_at=session.created_at,
        original_image=original_image,
        enhanced_image=enhanced_image,
    )
