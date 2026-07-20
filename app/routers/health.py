from datetime import datetime, timezone

from fastapi import APIRouter

from app.models.classifier import classifier
from app.models.enhancer import enhancer
from app.schemas.responses import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        classifier_loaded=classifier.model is not None,
        enhancer_loaded=enhancer.model is not None,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
