# HUMOS — Navigation Ground Truth from Video

AI-powered pipeline that converts raw video into structured, navigation-relevant ground truth for humanoid robots. Detects objects, tracks them across frames, classifies navigation states (open/closed/blocked), and exports temporal ground truth with accuracy evaluation.

## Quick Start

```bash
uv sync
streamlit run app.py
```

If `uv` is not installed:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
streamlit run app.py
```

Requires GPU with >=8GB VRAM (sequential offload) or >=16GB (all models loaded).

![Architecture Diagram](docs/architecture_diagram.png)

## Features

- **Object tracking** with persistent IDs across frames
- **Zero-shot state classification** via SigLIP — 200x faster than VLM for open/closed/ajar labels
- **Navigation state classification** via VLM (doors, drawers, handles → open/closed/ajar/blocked)
- **Temporal diff** — VLM compares consecutive frames to detect state transitions
- **Navigation timeline** — per-object state timeline with colored bars and transition events
- **Hand-object interactions** via MediaPipe
- **Ground truth export** — structured JSON with per-frame annotations
- **Accuracy evaluation** — compare predictions against manual labels
- **Live perception** — real-time webcam inference with auto-recording and post-analysis
- **H.264 video export** — browser-playable annotated videos with in-app preview
- **Per-frame timing** — inference latency breakdown per model stage

## Pages

| Page | Description |
|---|---|
| `app.py` | Upload video → run multi-model pipeline → browse results → export ground truth |
| `pages/live.py` | Live webcam stream via WebRTC → real-time overlay → auto-record → post-stop analysis |

## Models

| Category | Model | Sizes | Notes |
|---|---|---|---|
| Detection | YOLOv8 | n/s/m/l/x | Ultralytics |
| Detection | YOLO11 | n/s/m/l/x | Ultralytics |
| Detection | RF-DETR | n/s/b/m/l | DINOv2 backbone, real-time transformer |
| Segmentation | SAM 3 | — | Text-prompted, streaming video sessions |
| Segmentation | FastSAM | — | Ultralytics, segment-everything mode |
| Segmentation | RF-DETR Seg | s/m/l | Instance segmentation with COCO classes |
| Tracking | YOLO Tracker | n/s/m/l/x | ByteTrack/BotSORT with persistent IDs |
| State Classifier | SigLIP | — | Zero-shot state classification (~50ms/crop vs ~10s VLM) |
| VLM | LFM2.5-VL | — | Transformers backend (CUDA/CPU) |
| VLM | LFM2.5-VL ONNX | — | ONNX Runtime (fp16 encoder + q4 decoder) |
| VLM | LFM2.5-VL MLX | — | Apple Silicon (mlx-vlm, 8-bit) |

## Architecture

```
core/             — types, ABCs, registry, video I/O
models/           — one file per model variant, registered via @register
  detection/      — yolov8, yolo11, rf-detr
  segmentation/   — sam3, fastsam, rf-detr-seg
  vlm/            — lfm2.5-vl (transformers, onnx, mlx backends in one file)
  tracking/       — yolo-tracker (bytetrack, botsort)
  state/          — siglip zero-shot classifier
pipeline/
  runner.py       — frame-by-frame orchestration, parallel inference, timing
  visualizer.py   — OpenCV drawing for all modalities
  interactions.py — MediaPipe hand-object interaction detection
  nav_state.py    — builds NavigationTimeline from tracker + VLM/state results
  exporter.py     — structured JSON ground truth export
  evaluator.py    — accuracy measurement against manual labels
app.py            — Streamlit UI (upload workflow)
pages/live.py     — Streamlit UI (live webcam workflow)
```

## Adding a Model

1. Create `models/<category>/<name>.py`
2. Implement the ABC with `load()`, `predict()`/`update()`/`classify()`, `unload()`
3. Decorate with `@register("category", "name")` and import in `models/__init__.py`

```python
from core.base import Detector
from core.registry import register

@register("detector", "my-detector")
class MyDetector(Detector):
    VALID_SIZES = ("s", "l")  # optional, enables size selector in UI
    def __init__(self, model_size="s"): ...
    def load(self): ...
    def predict(self, frame): ...
    def unload(self): ...
```

## Cluster Deployment (Singularity + SLURM)

Two container images:

| Image | Base | GPU Support |
|---|---|---|
| `world2data.sif` | `nvidia/cuda:12.4.1-runtime` | CUDA |
| `world2data-trt.sif` | `nvcr.io/nvidia/tensorrt:24.12-py3` | CUDA + TensorRT |

```bash
# Build image (after dep changes)
singularity build --fakeroot world2data.sif world2data.def
# or TensorRT variant:
singularity build --fakeroot world2data-trt.sif world2data-trt.def

# Deploy
source .env && scp -r . ${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_PATH}

# Submit
sbatch start_app.sbatch        # standard
sbatch start_app_trt.sbatch    # TensorRT

# Tunnel to access UI
ssh -L 8501:localhost:8501 -J ${CLUSTER_USER}@${CLUSTER_HOST} -p 7665 ${CLUSTER_USER}@<compute-node>
```

Source code is bind-mounted at runtime — no rebuild needed for code changes, only for dependency changes.

## Environment

Copy `.env.example` to `.env` and fill in:

```
HF_TOKEN=...           # HuggingFace token (for model downloads)
CLUSTER_HOST=...       # SLURM cluster hostname
CLUSTER_USER=...       # SSH user
CLUSTER_PATH=...       # Remote working directory
SSH_KEY_PATH=...       # SSH key for cluster
```

See [CLAUDE.md](CLAUDE.md) for full architecture details, compatibility notes, and development guidelines.
