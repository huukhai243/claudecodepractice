# cvdet — Object Detection Pipeline

Pipeline object detection hoàn chỉnh dựa trên YOLOv8: train → đánh giá → phục vụ qua REST API.
Cấu hình mặc định nhắm vào máy **chỉ có CPU**.

## Cài đặt

```bash
pip install -r requirements.txt
pip install -e .          # để `python -m cvdet.*` chạy được từ mọi thư mục
python scripts/check_env.py
```

Không muốn cài editable thì đặt `PYTHONPATH=src` trước mỗi lệnh.

Torch bản CPU cài riêng nếu máy chưa có:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Sử dụng

### Train

```bash
python -m cvdet.train                                   # dùng configs/default.yaml
python -m cvdet.train --dataset coco8.yaml --epochs 3   # chạy thử nhanh (~3 phút)
```

Dataset tự tải về `data/`. Checkpoint lưu ở `runs/<name>/weights/best.pt`.

### Đánh giá

```bash
python -m cvdet.evaluate --name smoke-coco8 --out outputs/metrics.json
```

In ra mAP50, mAP50-95, precision, recall, kèm mAP theo từng lớp.

### Dự đoán

```bash
python -m cvdet.predict anh.jpg --name smoke-coco8 --save outputs/ketqua.jpg
```

In JSON các box; `--save` ghi thêm ảnh đã vẽ box.

### API service

```bash
uvicorn cvdet.api.main:app --port 8000
```

Tài liệu tương tác: <http://127.0.0.1:8000/docs>

| Endpoint | Method | Trả về |
|---|---|---|
| `/health` | GET | Trạng thái service + model đang load |
| `/classes` | GET | Danh sách lớp nhận diện được |
| `/predict` | POST | JSON các box, nhãn, confidence |
| `/predict/image` | POST | Chính ảnh đó đã vẽ box (JPEG) |

```bash
curl -X POST -F "file=@anh.jpg" http://127.0.0.1:8000/predict
```

Service ưu tiên `runs/<name>/weights/best.pt`; chưa train thì dùng weights pretrained.

## Cấu hình

Mọi tham số nằm trong [configs/default.yaml](configs/default.yaml), ghi đè được bằng cờ dòng lệnh:

```bash
python -m cvdet.train --epochs 50 --imgsz 640 --batch 16
```

### Dataset

| Dataset | Ảnh | Dung lượng | Dùng khi |
|---|---|---|---|
| `coco8.yaml` | 8 | ~1 MB | Kiểm tra pipeline (~3 phút CPU) |
| `coco128.yaml` | 128 | ~7 MB | Mặc định |
| `VOC.yaml` | ~21k | ~2 GB | Train nghiêm túc, 20 lớp |
| `coco.yaml` | ~118k | ~20 GB | Cần GPU |

Dùng dataset riêng: tạo file YAML theo [format của Ultralytics](https://docs.ultralytics.com/datasets/detect/)
rồi trỏ `--dataset duong/dan/cua-ban.yaml`. Nhãn theo chuẩn YOLO — mỗi ảnh một file `.txt`,
mỗi dòng là `class_id x_center y_center width height`, toạ độ chuẩn hoá về [0, 1].

## Cấu trúc

```
configs/default.yaml     tham số
src/cvdet/
  config.py              load YAML + merge cờ CLI
  console.py             ép stdout UTF-8 (console Windows mặc định cp1252)
  train.py               fine-tune
  evaluate.py            đo chỉ số
  predict.py             lõi inference — dùng chung cho CLI và API
  api/main.py            FastAPI service
scripts/check_env.py     kiểm tra môi trường
tests/                   13 test, không cần tải weights
```

## Test

```bash
python -m pytest tests/ -q
```

Test API dùng detector giả nên không tải weights và chạy trong vài giây.

## Ghi chú về hiệu năng CPU

- Request đầu tiên tới API chậm (~4-5 giây) do model warm-up; các request sau nhanh hơn nhiều.
- Mixed precision (AMP) bị tắt khi chạy CPU — bật chỉ có lợi trên GPU.
- Train chậm thì giảm `--imgsz` hoặc `--epochs` trước khi nghĩ tới việc đổi model.
