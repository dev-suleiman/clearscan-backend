from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.auth import require_admin
from app.database import ScanSession, User, get_db
from app.routers.auth import profile
from app.storage import storage

router = APIRouter(prefix="/admin", tags=["admin"])


def user_summary(user: User) -> dict:
    return {**profile(user), "total_scans": len(user.sessions), "last_active": user.last_active.isoformat()}


@router.get("/users")
def users(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    return [user_summary(user) for user in db.query(User).filter(User.is_admin.is_(False)).all()]


@router.get("/users/{user_id}/history")
def user_history(user_id: int, _: User = Depends(require_admin), db: Session = Depends(get_db)):
    if db.get(User, user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")
    return [{"id": item.id, "user_id": item.user_id, "quality_class": item.quality_class,
             "enhancement_method": item.enhancement_method, "mode": item.mode,
             "created_at": item.created_at.isoformat(),
             "original_image_url": storage.get_signed_url(item.image_path) if item.image_path else None,
             "enhanced_image_url": storage.get_signed_url(item.enhanced_image_path) if item.enhanced_image_path else None}
            for item in db.query(ScanSession).filter(ScanSession.user_id == user_id)
            .order_by(ScanSession.created_at.desc()).all()]


@router.get("/stats")
def stats(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    now = datetime.utcnow()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week = now - timedelta(days=7)
    most_active = (db.query(User, func.count(ScanSession.id).label("count"))
                   .outerjoin(ScanSession).filter(User.is_admin.is_(False))
                   .group_by(User.id).order_by(func.count(ScanSession.id).desc()).first())
    return {"total_users": db.query(User).filter(User.is_admin.is_(False)).count(),
            "scans_today": db.query(ScanSession).filter(ScanSession.created_at >= today).count(),
            "scans_this_week": db.query(ScanSession).filter(ScanSession.created_at >= week).count(),
            "most_active_user": profile(most_active[0]) if most_active else None}