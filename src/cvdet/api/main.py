"""FastAPI service phục vụ model object detection.

Chạy:
    uvicorn cvdet.api.main:app --reload --port 8000
Tài liệu tương tác: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from ..config import load_config
from ..predict import Detector, detector_from_config
from .schemas import DetectionOut, ErrorResponse, HealthResponse, PredictResponse

# Giới hạn kích thước file để một upload lỗi không làm cạn RAM của service.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}

# State của app; điền trong lifespan để model chỉ load một lần khi khởi động.
_state: dict[str, Any] = {"detector": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    # Load ngay lúc khởi động, không lazy: request đầu tiên không phải chờ vài giây,
    # và nếu weights hỏng thì service fail nhanh thay vì lỗi lúc đang phục vụ.
    _state["detector"] = detector_from_config(cfg)
    yield
    _state["detector"] = None


app = FastAPI(
    title="cvdet — Object Detection API",
    description="Phát hiện vật thể trong ảnh bằng YOLOv8.",
    version="0.1.0",
    lifespan=lifespan,
)


def _get_detector() -> Detector:
    detector = _state["detector"]
    if detector is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng")
    return detector


async def _read_image(file: UploadFile) -> np.ndarray:
    """Đọc file upload thành mảng BGR cho OpenCV/YOLO. Ném HTTPException nếu không hợp lệ."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Định dạng '{file.content_type}' không hỗ trợ. "
            f"Chấp nhận: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="File rỗng")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Ảnh vượt quá giới hạn {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )

    from PIL import Image, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Không giải mã được ảnh") from exc

    # YOLO nhận numpy array theo thứ tự kênh BGR (quy ước của OpenCV).
    rgb = np.asarray(image.convert("RGB"))
    return rgb[:, :, ::-1].copy()


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Kiểm tra service và model đã sẵn sàng chưa."""
    detector = _state["detector"]
    if detector is None:
        return HealthResponse(
            status="loading", model_loaded=False, weights="", device="", num_classes=0
        )
    return HealthResponse(
        status="ok",
        model_loaded=True,
        weights=detector.weights,
        device=detector.device,
        num_classes=len(detector.class_names),
    )


@app.get("/classes", tags=["system"])
def classes() -> dict[int, str]:
    """Danh sách lớp model có thể nhận diện."""
    return _get_detector().class_names


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={400: {"model": ErrorResponse}, 415: {"model": ErrorResponse}},
    tags=["inference"],
)
async def predict(
    file: Annotated[UploadFile, File(description="Ảnh cần phân tích")],
    conf: Annotated[float | None, Query(ge=0.0, le=1.0, description="Ghi đè ngưỡng confidence")] = None,
) -> PredictResponse:
    """Trả về danh sách vật thể phát hiện được dưới dạng JSON."""
    detector = _get_detector()
    image = await _read_image(file)

    # Ghi đè ngưỡng cho riêng request này rồi khôi phục — detector dùng chung mọi request.
    original_conf = detector.conf
    if conf is not None:
        detector.conf = conf
    try:
        started = time.perf_counter()
        detections = detector.predict(image)
        elapsed_ms = (time.perf_counter() - started) * 1000
    finally:
        detector.conf = original_conf

    height, width = image.shape[:2]
    return PredictResponse(
        filename=file.filename or "upload",
        width=width,
        height=height,
        count=len(detections),
        inference_ms=round(elapsed_ms, 2),
        detections=[
            DetectionOut(
                class_id=d.class_id,
                label=d.label,
                confidence=d.confidence,
                box=list(d.box),
            )
            for d in detections
        ],
    )


@app.post(
    "/predict/image",
    responses={200: {"content": {"image/jpeg": {}}}, 415: {"model": ErrorResponse}},
    response_class=Response,
    tags=["inference"],
)
async def predict_image(
    file: Annotated[UploadFile, File(description="Ảnh cần phân tích")],
) -> Response:
    """Trả về chính ảnh đó đã được vẽ bounding box — tiện để mắt thường kiểm tra."""
    import cv2

    detector = _get_detector()
    image = await _read_image(file)
    annotated = detector.annotate(image)

    ok, buffer = cv2.imencode(".jpg", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Không mã hoá được ảnh kết quả")

    return Response(content=buffer.tobytes(), media_type="image/jpeg")
