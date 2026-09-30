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
    # Thư mục chứa dataset tải về, tương đối với gốc project.
    root: str = "data"


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


def configure_ultralytics_dirs(cfg: Config) -> Path:
    """Buộc Ultralytics tải dataset/weights vào trong project.

    Mặc định nó dùng `Path.home()/datasets`, nhưng dưới Git Bash trên Windows home
    có thể resolve thành `C:/Users` — thư mục không có quyền ghi, làm train fail.
    Ghim đường dẫn vào <project>/data cũng giúp dataset đi cùng project.

    Lưu ý: `settings.update` ghi vào file cấu hình chung của Ultralytics, nên thay
    đổi này có tác dụng ngoài phạm vi project.
    """
    from ultralytics import settings

    data_dir = (PROJECT_ROOT / cfg.data.root).resolve()
    weights_dir = data_dir / "weights"
    for path in (data_dir, weights_dir):
        path.mkdir(parents=True, exist_ok=True)

    updates = {}
    if settings.get("datasets_dir") != str(data_dir):
        updates["datasets_dir"] = str(data_dir)
    if settings.get("weights_dir") != str(weights_dir):
        updates["weights_dir"] = str(weights_dir)
    if updates:
        settings.update(updates)
        # settings.update chỉ ghi ra file; các module Ultralytics đã import xong thì
        # vẫn giữ hằng số cũ (`from ultralytics.utils import DATASETS_DIR`). Không
        # gán đè thì lần chạy ĐẦU TIÊN trên máy mới vẫn dùng đường dẫn cũ, chỉ từ
        # lần thứ hai mới đúng.
        _rebind_ultralytics_constants(data_dir, weights_dir)

    return data_dir


def _rebind_ultralytics_constants(data_dir: Path, weights_dir: Path) -> None:
    """Gán lại DATASETS_DIR / WEIGHTS_DIR trên mọi module ultralytics đã import."""
    import sys

    for name, module in list(sys.modules.items()):
        if not name.startswith("ultralytics"):
            continue
        if getattr(module, "DATASETS_DIR", None) is not None:
            module.DATASETS_DIR = data_dir
        if getattr(module, "WEIGHTS_DIR", None) is not None:
            module.WEIGHTS_DIR = weights_dir


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
