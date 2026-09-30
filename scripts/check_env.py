"""Kiểm tra môi trường trước khi train: package, thiết bị, dung lượng.

Chạy: python scripts/check_env.py
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
from pathlib import Path

# check_env.py phải chạy được trước khi cài package, nên không import cvdet.console
# mà lặp lại logic UTF-8 ở đây (console Windows mặc định là cp1252).
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        try:
            _reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

REQUIRED = ["torch", "torchvision", "ultralytics", "numpy", "PIL", "yaml"]
OPTIONAL = ["fastapi", "uvicorn", "cv2", "pytest"]


def _check(names: list[str], label: str) -> list[str]:
    print(f"\n{label}")
    missing = []
    for name in names:
        found = importlib.util.find_spec(name) is not None
        print(f"  [{'v' if found else 'x'}] {name}")
        if not found:
            missing.append(name)
    return missing


def main() -> int:
    print("=" * 52)
    print("  KIỂM TRA MÔI TRƯỜNG")
    print("=" * 52)
    print(f"\nPython : {sys.version.split()[0]}  ({platform.machine()})")
    print(f"OS     : {platform.system()} {platform.release()}")

    missing_required = _check(REQUIRED, "Package bắt buộc:")
    _check(OPTIONAL, "Package tuỳ chọn (cần cho API / test):")

    print("\nThiết bị tính toán:")
    if importlib.util.find_spec("torch") is None:
        print("  ? chưa cài torch")
    else:
        import torch

        print(f"  torch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  [v] CUDA: {torch.cuda.get_device_name(0)}")
            print("      -> Có thể tăng --imgsz 640 và --batch 16")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            print("  [v] Apple MPS khả dụng")
        else:
            print("  [!] Chỉ có CPU — train sẽ chậm.")
            print("      -> Giữ imgsz=416, batch=4, hoặc dùng dataset coco8.yaml để thử nhanh.")

    project_root = Path(__file__).resolve().parents[1]
    free_gb = shutil.disk_usage(project_root).free / (1024**3)
    print(f"\nDung lượng trống ({project_root.drive or project_root}): {free_gb:.1f} GB")
    if free_gb < 5:
        print("  [!] Dưới 5 GB — dataset + checkpoint có thể không đủ chỗ.")

    if missing_required:
        print(f"\n[x] Thiếu package bắt buộc: {', '.join(missing_required)}")
        print("    Cài bằng: pip install -r requirements.txt")
        return 1

    print("\n[v] Môi trường sẵn sàng. Bước tiếp: python -m cvdet.train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
