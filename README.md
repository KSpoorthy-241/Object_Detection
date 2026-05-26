# Hybrid Object Detector

A high-accuracy object detection system combining **YOLOv8** with a **ResNet50 FPN backbone**, served via a Streamlit web app.

## How It Works

Plain YOLOv8 is enhanced with four accuracy-boosting techniques:

| Technique | Effect |
|---|---|
| ResNet50 FPN backbone | Richer multi-scale feature extraction |
| CBAM attention | Focuses on relevant spatial regions and channels |
| Feature injection (gated) | ResNet50 context fused into YOLO's feature space |
| ROI-align rescoring | Per-box CNN quality score blended into YOLO confidence |
| Test-Time Augmentation (TTA) | Runs detection on flips and scales to catch missed objects |
| Weighted Box Fusion (WBF) | Merges TTA predictions more accurately than NMS |

## Project Structure

```
├── hybrid_detector.py   # Core detector logic
├── streamlit_app.py     # Web UI
├── yolov8n.pt           # YOLOv8 nano weights
├── yolov8x.pt           # YOLOv8 extra-large weights (used by default)
└── results/             # Output images
```

## Requirements

```bash
pip install ultralytics streamlit torch torchvision opencv-python pillow
```

## Usage

### Run the web app

```bash
streamlit run streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

### Modes

- **Image Upload** — upload a JPG/PNG and click "Run Detection"
- **Webcam** — live detection from your webcam feed

### Use the detector directly

```python
from hybrid_detector import HybridObjectDetector

detector = HybridObjectDetector(yolo_model='yolov8x.pt', use_cnn_features=True)
results, features, elapsed = detector.detect('image.jpg', conf=0.25, iou=0.45)
print(f"Detected {len(results.boxes)} objects in {elapsed:.2f}s")
```

## Settings

| Parameter | Default | Description |
|---|---|---|
| `yolo_model` | `yolov8x.pt` | YOLO weights file |
| `use_cnn_features` | `True` | Enable ResNet50 feature fusion |
| `rescore_alpha` | `0.35` | Blend weight for CNN rescoring (0 = YOLO only) |
| `use_tta` | `True` | Enable Test-Time Augmentation |

## Hardware

Runs on CPU or GPU. GPU is strongly recommended for real-time webcam use.
