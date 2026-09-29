"""Schema request/response của API — nguồn sự thật cho tài liệu OpenAPI."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DetectionOut(BaseModel):
    class_id: int = Field(..., description="Chỉ số lớp trong model", examples=[0])
    label: str = Field(..., description="Tên lớp", examples=["person"])
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.9123])
    box: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[x1, y1, x2, y2] theo pixel của ảnh gốc",
        examples=[[34.5, 12.0, 220.75, 410.25]],
    )


class PredictResponse(BaseModel):
    filename: str
    width: int
    height: int
    count: int = Field(..., description="Số vật thể phát hiện được")
    inference_ms: float = Field(..., description="Thời gian suy luận, mili giây")
    detections: list[DetectionOut]


class HealthResponse(BaseModel):
    # `model_loaded` trùng namespace `model_` được pydantic bảo lưu — tắt cảnh báo.
    model_config = ConfigDict(protected_namespaces=())

    status: str = "ok"
    model_loaded: bool
    weights: str
    device: str
    num_classes: int


class ErrorResponse(BaseModel):
    detail: str
