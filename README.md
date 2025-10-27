# Hot-Desk Office Item Classifier (Part B)

This repository contains the code, data pointers and notes for the Office-Goods classification component (Part B) of the PDE3802 coursework.

## 📦 Contents
- `src/demo_app.py` — Gradio demo app for inference.
- `data/data.yaml` — dataset configuration used for training.
- `models/hotdesk_final_model.pt` — trained YOLO11m weights (large file, add separately).
- `docs/DATASET_CARD.md` — dataset description.
- `docs/ERROR_ANALYSIS.md` — per-class and model error analysis.
- `docs/PARTNER_HANDOVER.md` — collaboration notes for integration with Part A.
- `docs/CHANGELOG.md` — project history.

## 🚀 Environment
- OS: Ubuntu 22.04.5 LTS
- GPU: AMD Radeon RX 6800 XT (16GB)
- ROCm: 6.0
- Python: 3.10.12
- PyTorch: 2.4.1+rocm6.0
- Ultralytics: 8.3.221

## 🧠 Training Overview
- Framework: YOLO11m (20.1M parameters)
- Dataset size: 8,345 labelled instances across 8 classes
- Phases:
  1. **Full dataset, heavy aug**
  2. **Fine-tune best weights**
  3. **'Mug' class micro fine-tune**

## 📊 Final Model Results
| Metric | Value |
|---------|--------|
| mAP50 | 0.939 |
| mAP50-95 | 0.758 |
| Precision | 0.935 |
| Recall | 0.909 |
| F1 Score | 0.922 |
| Macro-F1 | ≈ 0.920 |

Per-class AP50: mug 0.81, headset 0.85, mouse 0.94, stapler 0.99, notebook 0.99, pen 0.96, phone 0.99, bottle 0.98.

## 🧩 How to Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-rocm.txt
python src/demo_app.py
```

Results and plots are saved in `runs/detect/`.
