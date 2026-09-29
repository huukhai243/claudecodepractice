"""Train model object detection. Dataset công khai được Ultralytics tự tải về."""

from __future__ import annotations

import argparse
import time

from .config import Config, add_common_args, apply_overrides, load_config, resolve_device


def train(cfg: Config) -> str:
    """Train và trả về đường dẫn checkpoint tốt nhất."""
    from ultralytics import YOLO

    device = resolve_device(cfg.train.device)

    print("=" * 60)
    print(f"  Model    : {cfg.model.weights}")
    print(f"  Dataset  : {cfg.data.dataset}")
    print(f"  Device   : {device}")
    print(f"  Epochs   : {cfg.train.epochs}   imgsz: {cfg.train.imgsz}   batch: {cfg.train.batch}")
    print("=" * 60)

    if device == "cpu":
        print(
            "\n[!] Đang train trên CPU. Nếu quá chậm, giảm --epochs hoặc --imgsz,\n"
            "    hoặc đổi --dataset coco8.yaml (chỉ 8 ảnh) để kiểm tra pipeline.\n"
        )

    model = YOLO(cfg.model.weights)
    started = time.perf_counter()

    model.train(
        data=cfg.data.dataset,
        epochs=cfg.train.epochs,
        imgsz=cfg.train.imgsz,
        batch=cfg.train.batch,
        workers=cfg.train.workers,
        device=device,
        patience=cfg.train.patience,
        seed=cfg.train.seed,
        project=str(cfg.run_dir.parent),
        name=cfg.run_dir.name,
        exist_ok=True,
        # AMP chỉ có lợi trên GPU; bật trên CPU sẽ chậm hơn.
        amp=device != "cpu",
        plots=True,
    )

    elapsed = time.perf_counter() - started
    print(f"\nTrain xong sau {elapsed / 60:.1f} phút.")
    print(f"Kết quả       : {cfg.run_dir}")
    print(f"Checkpoint tốt: {cfg.best_weights}")
    return str(cfg.best_weights)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train model object detection")
    add_common_args(parser)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", type=str, default=None, help="Tên lần chạy (thư mục runs/)")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args)
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
