# Project Requirements: YOLO11n Object Detection

## Overview

This project leverages the YOLO11n model (the smallest and fastest variant in the Ultralytics YOLO11 family) for real-time object detection. The workflow includes running inference with the provided PyTorch model, converting it to ONNX, running inference with the ONNX model, and comparing results (including IoU analysis).

---

## 1. Input Files

- **Model:** `yolo11n.pt` (PyTorch format, provided by Ultralytics)
- **Image:** `image.png` (sample image for inference)

---

## 2. Environment Setup

- Use a fresh Python virtual environment (`.venv` recommended).
- All dependencies must be listed in `requirements.txt`.
- Required packages include (but are not limited to):
  - `torch`
  - `torchvision`
  - `onnx`
  - `onnxruntime`
  - `opencv-python`
  - `matplotlib`
  - `numpy`
  - Any other utility packages as needed

---

## 3. Inference with PyTorch Model

- Load `yolo11n.pt` using PyTorch.
- Run inference on `image.png`.
- Output:
  - Bounding box coordinates
  - Class labels
  - Labeled output image with detections (`outputs/pt_result.png`)
  - Human-readable summary (`outputs/summary.txt`)

---

## 4. Model Conversion: PyTorch to ONNX

- Convert `yolo11n.pt` to ONNX format.
- Validate the ONNX model by running inference on `image.png`.
- Output:
  - Labeled output image with detections from ONNX model (`outputs/onnx_result.png`)
  - Human-readable summary (append to `outputs/summary.txt`)

---

## 5. IoU Comparison (Optional but Recommended)

- Compare bounding boxes from PyTorch and ONNX inferences.
- Calculate Intersection over Union (IoU) for each detected object.
- Visualize IoU comparison:
  - IoU chart (`outputs/iou_chart.png`)
  - IoU summary (`outputs/iou_summary.txt`)

---

## 6. Output Structure

All outputs must be saved in the `outputs/` directory:
- `pt_result.png`: PyTorch inference result image
- `onnx_result.png`: ONNX inference result image
- `summary.txt`: Human-readable summary of results
- `iou_chart.png`: IoU comparison chart
- `iou_summary.txt`: IoU statistics and analysis

---

## 7. Project Structure

- `inputs/`: Input images
- `models/`: Model files (`yolo11n.pt`, ONNX model)
- `outputs/`: All output files
- `src/`: All scripts and logic
  - `main.py`: Main entry point for running the project
- `docs/`: Documentation
  - `requirement.md`: This file
  - `plan.md`: Project plan and internal task tracking
- `.venv/`: Python virtual environment
- `requirements.txt`: Python dependencies
- `run.sh`: Bash script to install requirements and run the project

---

## 8. Automation

- Provide a `run.sh` script that:
  - Installs all dependencies from `requirements.txt`
  - Runs the main project script (`src/main.py`)

---

## 9. AI-Assisted Coding & Documentation

- Use AI-assisted coding tools (e.g., Windsurf, Cursor Pro) to expedite development.
- Document how AI tools were used in the process (in video or summary).

---

## 10. Deliverables

- All code, scripts, and outputs as described above.
- A video walkthrough (screen + audio) explaining the process, environment setup, and results.
- Clear, human-readable summaries and visualizations of results.

---

## 11. Evaluation Criteria

- **Completeness:** Both PyTorch and ONNX inference must be demonstrated.
- **Environment Setup:** Clean, reproducible setup.
- **AI Tool Usage:** Effective use of AI-assisted coding.
- **Clarity:** Clear explanations and documentation.
- **Correctness:** Sensible and accurate outputs (bounding boxes, labels, IoU, etc.).

---
