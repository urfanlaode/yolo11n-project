# YOLO11n-Project: Project Plan & TODO List

## Overview

This project aims to demonstrate object detection using the YOLO11n model (nano variant from Ultralytics), including:
- Running inference on a sample image using the provided PyTorch model.
- Converting the PyTorch model to ONNX and running inference with ONNX.
- Comparing results, including IoU analysis.
- Providing a reproducible, well-documented, and automated workflow.

---

## Project Structure

- `inputs/` — Input images for inference.
- `models/` — Model files (PyTorch, ONNX).
- `src/` — Source code and scripts.
- `outputs/` — Output images, results, and summaries.
- `docs/` — Documentation, requirements, and plans.

---

## TODO List

### 1. Environment Setup
- [x] Create and activate a Python virtual environment (`.venv`)
- [x] Prepare `requirements.txt` with all necessary packages (torch, torchvision, onnx, onnxruntime, numpy, matplotlib, etc.)
- [x] Write `run.sh` to automate environment setup and project execution

### 2. Data & Model Preparation
- [x] Move `image.png` to `inputs/`
- [x] Move `yolo11n.pt` to `models/`
- [x] Download or generate ONNX model after conversion

### 3. PyTorch Inference
- [x] Implement script to load YOLO11n PyTorch model
- [x] Run inference on `inputs/image.png`
- [x] Save detection results (bounding boxes, labels) and output image to `outputs/pt_result.png`
- [x] Save human-readable summary to `outputs/summary.txt`

### 4. Model Conversion: PyTorch to ONNX
- [x] Implement script to convert `models/yolo11n.pt` to ONNX format
- [x] Validate ONNX model

### 5. ONNX Inference
- [x] Implement script to run inference with ONNX model on `inputs/image.png`
- [x] Save detection results and output image to `outputs/onnx_result.png`
- [x] Update `outputs/summary.txt` with ONNX results

### 6. IoU Comparison & Visualization
- [x] Implement IoU calculation between PyTorch and ONNX detections
- [x] Generate IoU chart and save to `outputs/iou_chart.png`
- [x] Save IoU summary to `outputs/iou_summary.txt`

### 7. Documentation
- [x] Write `docs/requirement.md` (structured requirements)
- [x] Update `docs/plan.md` (this file) as tasks progress

### 8. Automation & Usability
- [x] Ensure `src/main.py` acts as the main entry point for all logic
- [x] Ensure `run.sh` installs requirements and runs the project end-to-end

### 9. (Optional) Video Walkthrough
- [x] Record screen and audio while setting up and running the project
- [x] Explain use of AI-assisted coding tools

---

## Notes

- All outputs must be saved under `outputs/` as specified.
- Code should be modular, clear, and well-commented.
- Follow best practices for reproducibility and documentation.
- Use AI-assisted tools (e.g., Cursor, Windsurf) where possible and document their usage.

---

## Progress Tracking

- [x] Initial setup complete
- [x] PyTorch inference working
- [x] ONNX conversion and inference working
- [x] IoU analysis complete
- [x] Documentation finalized
- [x] Automation script tested

---
