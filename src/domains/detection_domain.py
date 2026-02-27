import os

import numpy as np

from utils.image_utils import load_image, save_detection_image
from utils.iou_utils import compare_iou, compare_iou_matrix
from utils.model_utils import (
    load_pytorch_model,
    run_onnx_inference,
    run_pytorch_inference,
)
from utils.viz_utils import plot_iou_chart

COCO_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

COLOR_PALETTE = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (64, 0, 0),
    (0, 64, 0),
    (0, 0, 64),
    (64, 64, 0),
    (64, 0, 64),
    (0, 64, 64),
    (192, 0, 0),
    (0, 192, 0),
    (0, 0, 192),
    (192, 192, 0),
    (192, 0, 192),
    (0, 192, 192),
]


class DetectionDomain:
    def __init__(
        self,
        input_image_path,
        pt_model_path,
        onnx_model_path,
        output_pt_image,
        output_onnx_image,
        output_summary,
        output_iou_chart,
        output_iou_summary,
    ):
        self.input_image_path = input_image_path
        self.pt_model_path = pt_model_path
        self.onnx_model_path = onnx_model_path
        self.output_pt_image = output_pt_image
        self.output_onnx_image = output_onnx_image
        self.output_summary = output_summary
        self.output_iou_chart = output_iou_chart
        self.output_iou_summary = output_iou_summary

        os.makedirs(os.path.dirname(self.output_pt_image), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_onnx_image), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_summary), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_iou_chart), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_iou_summary), exist_ok=True)

    def run(self):
        # Load input image
        orig_img, img_tensor, orig_shape, resized_shape = load_image(
            self.input_image_path
        )

        # PyTorch inference
        pt_model = load_pytorch_model(self.pt_model_path)
        pt_boxes, pt_scores, pt_labels = run_pytorch_inference(
            pt_model, self.input_image_path
        )
        save_detection_image(
            orig_img,
            pt_boxes,
            pt_scores,
            pt_labels,
            self.output_pt_image,
            class_names=COCO_NAMES,
            colors=COLOR_PALETTE,
        )

        # Convert to ONNX if needed
        if not os.path.exists(self.onnx_model_path):
            pt_model.export(
                format="onnx", optimize=True, imgsz=img_tensor.shape[-1], device="cpu"
            )

        # ONNX inference
        onnx_boxes, onnx_scores, onnx_labels = run_onnx_inference(
            self.onnx_model_path, self.input_image_path
        )
        save_detection_image(
            orig_img,
            onnx_boxes,
            onnx_scores,
            onnx_labels,
            self.output_onnx_image,
            class_names=COCO_NAMES,
            colors=COLOR_PALETTE,
        )

        # IoU comparison and chart
        iou_scores = compare_iou(pt_boxes, onnx_boxes)
        iou_matrix = compare_iou_matrix(pt_boxes, onnx_boxes)
        if iou_matrix.size > 0:
            plot_iou_chart(
                iou_scores,
                self.output_iou_chart,
                pt_scores=pt_scores,
                onnx_scores=onnx_scores,
            )

        # Write summary
        with open(self.output_summary, "w") as f:
            f.write("YOLO11n-Project: Inference Summary\n\n")
            f.write("PyTorch Model Detections:\n")
            for i, (b, s, l) in enumerate(zip(pt_boxes, pt_scores, pt_labels)):
                label_str = (
                    COCO_NAMES[int(l)] if 0 <= int(l) < len(COCO_NAMES) else str(int(l))
                )
                f.write(f"  [{i}] Box: {b}, Score: {s:.2f}, Label: {label_str}\n")
            f.write("\nONNX Model Detections:\n")
            for i, (b, s, l) in enumerate(zip(onnx_boxes, onnx_scores, onnx_labels)):
                label_str = (
                    COCO_NAMES[int(l)] if 0 <= int(l) < len(COCO_NAMES) else str(int(l))
                )
                f.write(f"  [{i}] Box: {b}, Score: {s:.2f}, Label: {label_str}\n")

        with open(self.output_iou_summary, "w") as f:
            f.write("IoU Scores (PyTorch vs ONNX):\n")
            for i, score in enumerate(iou_scores):
                f.write(f"Detection {i}: IoU = {score:.3f}\n")
            f.write(f"\nMean IoU: {np.mean(iou_scores):.3f}\n")
