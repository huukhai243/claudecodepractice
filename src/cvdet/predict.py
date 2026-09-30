"""Inference: load model một lần, chạy dự đoán trên ảnh.

Module này là lõi dùng chung cho cả CLI (`python -m cvdet.predict`) và API service,
nên nó không phụ thuộc vào FastAPI hay argparse.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import Config, add_common_args, apply_overrides, load_config, resolve_device
from .console import setup_console


@dataclass(frozen=True)
class Detection:
    """Một vật thể được phát hiện. Toạ độ theo pixel của ảnh gốc."""

    class_id: int
    label: str
    confidence: float
    # [x1, y1, x2, y2] — góc trên-trái và dưới-phải
    box: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Detector:
    """Bọc model YOLO. Khởi tạo tốn vài giây nên hãy tái sử dụng instance."""

    def __init__(
        self,
        weights: str | Path,
        device: str = "auto",
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 100,
        imgsz: int = 416,
    ) -> None:
        # Import trong hàm để `import cvdet.predict` không kéo theo torch khi chưa cần.
        from ultralytics import YOLO

        self.weights = str(weights)
        self.device = resolve_device(device)
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.imgsz = imgsz
        self.model = YOLO(self.weights)
        # names: {class_id: tên lớp}
        self.class_names: dict[int, str] = dict(self.model.names)

    def predict(self, source: str | Path | np.ndarray) -> list[Detection]:
        """Chạy dự đoán trên một ảnh (đường dẫn hoặc mảng numpy BGR)."""
        results = self.model.predict(
            source=source,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            imgsz=self.imgsz,
            verbose=False,
        )
        if not results:
            return []
        return self._parse(results[0])

    def predict_batch(
        self, sources: Sequence[str | Path | np.ndarray]
    ) -> list[list[Detection]]:
        """Dự đoán nhiều ảnh trong một lần gọi — nhanh hơn gọi lặp `predict`."""
        if not sources:
            return []
        results = self.model.predict(
            source=list(sources),
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            imgsz=self.imgsz,
            verbose=False,
        )
        return [self._parse(r) for r in results]

    def annotate(self, source: str | Path | np.ndarray) -> np.ndarray:
        """Trả về ảnh BGR đã vẽ sẵn box + nhãn (dùng cho endpoint /predict/image)."""
        results = self.model.predict(
            source=source,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            imgsz=self.imgsz,
            verbose=False,
        )
        if not results:
            raise ValueError("Không đọc được ảnh đầu vào")
        return results[0].plot()

    def _parse(self, result: Any) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        return [
            Detection(
                class_id=int(cid),
                label=self.class_names.get(int(cid), str(cid)),
                confidence=round(float(score), 4),
                box=(
                    round(float(x1), 2),
                    round(float(y1), 2),
                    round(float(x2), 2),
                    round(float(y2), 2),
                ),
            )
            for (x1, y1, x2, y2), score, cid in zip(xyxy, confs, class_ids)
        ]


@lru_cache(maxsize=2)
def get_detector(
    weights: str,
    device: str,
    conf: float,
    iou: float,
    max_det: int,
    imgsz: int,
) -> Detector:
    """Detector được cache theo tham số — API service dùng hàm này để load 1 lần."""
    return Detector(
        weights=weights, device=device, conf=conf, iou=iou, max_det=max_det, imgsz=imgsz
    )


def detector_from_config(cfg: Config) -> Detector:
    """Ưu tiên checkpoint đã train (best.pt); chưa train thì dùng weights pretrained."""
    weights = cfg.best_weights if cfg.best_weights.exists() else Path(cfg.model.weights)
    return get_detector(
        weights=str(weights),
        device=cfg.train.device,
        conf=cfg.predict.conf,
        iou=cfg.predict.iou,
        max_det=cfg.predict.max_det,
        imgsz=cfg.train.imgsz,
    )


def main() -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Chạy object detection trên ảnh")
    add_common_args(parser)
    parser.add_argument("source", type=str, help="Đường dẫn ảnh, thư mục ảnh, hoặc URL")
    parser.add_argument("--conf", type=float, default=None, help="Ngưỡng confidence")
    parser.add_argument("--iou", type=float, default=None, help="Ngưỡng IoU cho NMS")
    parser.add_argument("--max-det", dest="max_det", type=int, default=None)
    parser.add_argument("--name", type=str, default=None, help="Tên lần chạy (thư mục runs/) để lấy best.pt")
    parser.add_argument(
        "--save", type=Path, default=None, help="Lưu ảnh đã vẽ box vào đường dẫn này"
    )
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args)
    detector = detector_from_config(cfg)

    detections = detector.predict(args.source)
    print(json.dumps([d.to_dict() for d in detections], indent=2, ensure_ascii=False))

    if args.save:
        import cv2

        args.save.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save), detector.annotate(args.source))
        print(f"\nĐã lưu ảnh chú thích: {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
