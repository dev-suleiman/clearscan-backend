from pydantic import BaseModel


class MetricsDict(BaseModel):
    laplacian_variance: float
    rms_contrast: float
    shannon_entropy: float
    histogram_spread: float
    noise_variance: float
    # Reference-based metrics (enhanced vs. original). Not meaningful for a
    # standalone "before" reading, so left unset there.
    ssim: float | None = None
    psnr: float | None = None


class QualityAssessmentResponse(BaseModel):
    quality_class: str
    confidence: float
    class_probabilities: dict[str, float]
    defects: list[str]
    metrics: MetricsDict
    processing_time_ms: float
    mode: str


class EnhancementResponse(BaseModel):
    enhanced_image_b64: str
    method: str
    before_metrics: MetricsDict
    after_metrics: MetricsDict
    processing_time_ms: float


class ComparisonResponse(BaseModel):
    winner: str
    winning_image_b64: str
    clahe_result: dict
    cnn_result: dict
    composite_scores: dict
    processing_time_ms: float
    fallback_reason: str | None = None


class HealthResponse(BaseModel):
    status: str
    classifier_loaded: bool
    enhancer_loaded: bool
    timestamp: str
