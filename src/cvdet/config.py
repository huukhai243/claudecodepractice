"""Đọc và hợp nhất cấu hình từ YAML + override dòng lệnh."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Thư mục gốc của project (…/claudecodepractice), tính ngược từ file này.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass
class ModelConfig:
    weights: str = "yolov8n.pt"


@dataclass
class DataConfig:
    dataset: str = "coco128.yaml"


@dataclass
class TrainConfig:
    epochs: int = 10
    imgsz: int = 416
    batch: int = 4
    workers: int = 2
    device: str = "auto"
    patience: int = 20
    seed: int = 42
    project: str = "runs"
    name: str = "yolov8n-coco128"


@dataclass
class PredictConfig:
    conf: float = 0.25
    iou: float = 0.45
    max_det: int = 100


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)

    @property
    def run_dir(self) -> Path:
        """Thư mục Ultralytics ghi kết quả của lần chạy hiện tại."""
        return PROJECT_ROOT / self.train.project / self.train.name

    @property
    def best_weights(self) -> Path:
        """Checkpoint tốt nhất sau khi train."""
        return self.run_dir / "weights" / "best.pt"


def _build_section(section_cls: type, raw: dict[str, Any] | None) -> Any:
    """Khởi tạo dataclass, bỏ qua khoá lạ để YAML thừa field không làm crash."""
    raw = raw or {}
    known = {f.name for f in section_cls.__dataclass_fields__.values()}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"Khoá không hợp lệ trong mục '{section_cls.__name__}': {sorted(unknown)}. "
            f"Các khoá hợp lệ: {sorted(known)}"
        )
    return section_cls(**raw)


def load_config(path: str | Path | None = None) -> Config:
    """Đọc file YAML thành object Config. Thiếu file thì dùng giá trị mặc định."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return Config()

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    return Config(
        model=_build_section(ModelConfig, raw.get("model")),
        data=_build_section(DataConfig, raw.get("data")),
        train=_build_section(TrainConfig, raw.get("train")),
        predict=_build_section(PredictConfig, raw.get("predict")),
    )


def resolve_device(device: str = "auto") -> str:
    """Chuyển 'auto' thành thiết bị thực tế. Trả về chuỗi Ultralytics hiểu được."""
    if device != "auto":
        return device

    import torch

    if torch.cuda.is_available():
        return "0"
    # Apple Silicon
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Ghi đè config bằng các cờ dòng lệnh được truyền (bỏ qua cờ None)."""
    for section_name in ("model", "data", "train", "predict"):
        section = getattr(cfg, section_name)
        for field_name in section.__dataclass_fields__:
            value = getattr(args, field_name, None)
            if value is not None:
                setattr(section, field_name, value)
    return cfg


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Thêm các cờ override dùng chung cho mọi entrypoint."""
    parser.add_argument("--config", type=Path, default=None, help="Đường dẫn file YAML cấu hình")
    parser.add_argument("--weights", type=str, default=None, help="Checkpoint khởi tạo (.pt)")
    parser.add_argument("--dataset", type=str, default=None, help="File data YAML của dataset")
    parser.add_argument("--device", type=str, default=None, help="auto | cpu | 0 | mps")
    parser.add_argument("--imgsz", type=int, default=None, help="Kích thước ảnh đầu vào")
    return parser
