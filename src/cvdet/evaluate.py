"""Đánh giá model trên tập validation: mAP50, mAP50-95, precision, recall."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import (
    Config,
    add_common_args,
    apply_overrides,
    configure_ultralytics_dirs,
    load_config,
    resolve_device,
)
from .console import setup_console


def evaluate(cfg: Config, weights: Path | None = None) -> dict[str, Any]:
    """Chạy validation và trả về dict các chỉ số chính."""
    from ultralytics import YOLO

    if weights is None:
        weights = cfg.best_weights if cfg.best_weights.exists() else Path(cfg.model.weights)

    if not cfg.best_weights.exists() and weights == Path(cfg.model.weights):
        print(
            f"[!] Chưa tìm thấy {cfg.best_weights} — đang đánh giá weights pretrained "
            f"'{cfg.model.weights}'. Chạy `python -m cvdet.train` trước để có model của bạn.\n"
        )

    device = resolve_device(cfg.train.device)
    configure_ultralytics_dirs(cfg)
    model = YOLO(str(weights))

    metrics = model.val(
        data=cfg.data.dataset,
        imgsz=cfg.train.imgsz,
        batch=cfg.train.batch,
        device=device,
        verbose=False,
        # Không truyền project/name thì Ultralytics ghi vào ~/runs/detect/val,
        # tức là ra ngoài project. Giữ mọi kết quả cạnh lần train tương ứng.
        project=str(cfg.run_dir.parent),
        name=f"{cfg.train.name}-val",
        exist_ok=True,
    )

    box = metrics.box
    summary = {
        "weights": str(weights),
        "dataset": cfg.data.dataset,
        "mAP50": round(float(box.map50), 4),
        "mAP50_95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
    }

    # mAP50-95 theo từng lớp, sắp xếp giảm dần để thấy ngay lớp nào yếu.
    per_class = {
        model.names[int(cid)]: round(float(box.maps[int(cid)]), 4)
        for cid in getattr(box, "ap_class_index", [])
    }
    summary["per_class_mAP50_95"] = dict(
        sorted(per_class.items(), key=lambda kv: kv[1], reverse=True)
    )

    return summary


def main() -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Đánh giá model object detection")
    add_common_args(parser)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--name", type=str, default=None, help="Tên lần chạy (thư mục runs/) để lấy best.pt")
    parser.add_argument(
        "--out", type=Path, default=None, help="Ghi kết quả ra file JSON"
    )
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args)
    summary = evaluate(cfg)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nĐã lưu: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
