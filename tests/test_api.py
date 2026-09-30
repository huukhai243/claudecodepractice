"""Test API bằng detector giả — không load model thật nên chạy nhanh."""

from __future__ import annotations

import io

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("PIL")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from cvdet.api import main as api_main  # noqa: E402
from cvdet.predict import Detection  # noqa: E402


class FakeDetector:
    weights = "fake.pt"
    device = "cpu"
    conf = 0.25
    class_names = {0: "person", 1: "bicycle"}

    def predict(self, source):
        return [
            Detection(class_id=0, label="person", confidence=0.91, box=(1.0, 2.0, 3.0, 4.0))
        ]

    def annotate(self, source):
        return np.zeros((8, 8, 3), dtype=np.uint8)


@pytest.fixture
def client(monkeypatch):
    """TestClient với detector giả.

    Phải thay `detector_from_config` TRƯỚC khi vào context: TestClient chạy lifespan,
    và lifespan thật sẽ gọi YOLO() -> tải yolov8n.pt từ internet, làm test treo.
    """
    monkeypatch.setattr(api_main, "detector_from_config", lambda cfg: FakeDetector())
    with TestClient(api_main.app) as c:
        yield c


@pytest.fixture
def jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), color=(120, 60, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health_reports_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["num_classes"] == 2


def test_classes_endpoint(client):
    assert client.get("/classes").json() == {"0": "person", "1": "bicycle"}


def test_predict_returns_detections_and_image_size(client, jpeg_bytes):
    response = client.post(
        "/predict", files={"file": ("a.jpg", jpeg_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["width"] == 32 and body["height"] == 24
    assert body["detections"][0]["label"] == "person"


def test_predict_rejects_unsupported_type(client):
    response = client.post(
        "/predict", files={"file": ("a.txt", b"khong phai anh", "text/plain")}
    )
    assert response.status_code == 415


def test_predict_rejects_corrupt_image(client):
    response = client.post(
        "/predict", files={"file": ("a.jpg", b"day khong phai jpeg", "image/jpeg")}
    )
    assert response.status_code == 400


def test_conf_override_is_restored_after_request(client, jpeg_bytes):
    detector = api_main._state["detector"]
    before = detector.conf
    client.post(
        "/predict?conf=0.9", files={"file": ("a.jpg", jpeg_bytes, "image/jpeg")}
    )
    assert detector.conf == before


def test_predict_image_returns_jpeg(client, jpeg_bytes):
    pytest.importorskip("cv2")
    response = client.post(
        "/predict/image", files={"file": ("a.jpg", jpeg_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
