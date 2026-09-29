"""Test cho lớp config — chạy được mà không cần cài torch/ultralytics."""

from __future__ import annotations

import argparse

import pytest

from cvdet.config import Config, apply_overrides, load_config


def test_load_default_config():
    cfg = load_config()
    assert cfg.model.weights.endswith(".pt")
    assert cfg.train.epochs > 0
    assert 0.0 <= cfg.predict.conf <= 1.0


def test_missing_file_falls_back_to_defaults(tmp_path):
    cfg = load_config(tmp_path / "khong-ton-tai.yaml")
    assert cfg == Config()


def test_unknown_key_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("train:\n  epochsss: 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="epochsss"):
        load_config(path)


def test_cli_overrides_win_over_yaml():
    cfg = load_config()
    args = argparse.Namespace(epochs=99, imgsz=None, conf=0.9)
    cfg = apply_overrides(cfg, args)
    assert cfg.train.epochs == 99
    assert cfg.predict.conf == 0.9


def test_none_override_keeps_yaml_value():
    original = load_config().train.imgsz
    cfg = apply_overrides(load_config(), argparse.Namespace(imgsz=None))
    assert cfg.train.imgsz == original


def test_best_weights_path_under_run_dir():
    cfg = load_config()
    assert cfg.best_weights.parent.parent == cfg.run_dir
    assert cfg.best_weights.name == "best.pt"
