# This script delegates the workflow to the DetectionDomain.

# Paths
import glob
import os

from domains.detection_domain import DetectionDomain

# Dynamically find image.* in inputs directory
image_candidates = glob.glob(os.path.join("inputs", "image.*"))
if image_candidates:
    INPUT_IMAGE_PATH = image_candidates[0]
else:
    INPUT_IMAGE_PATH = None  # Or handle error appropriately
PT_MODEL_PATH = os.path.join("models", "yolo11n.pt")
ONNX_MODEL_PATH = os.path.join("models", "yolo11n.onnx")
OUTPUT_PT_IMAGE = os.path.join("outputs", "pt_result.png")
OUTPUT_ONNX_IMAGE = os.path.join("outputs", "onnx_result.png")
OUTPUT_SUMMARY = os.path.join("outputs", "summary.txt")
OUTPUT_IOU_CHART = os.path.join("outputs", "iou_chart.png")
OUTPUT_IOU_SUMMARY = os.path.join("outputs", "iou_summary.txt")

os.makedirs("outputs", exist_ok=True)


def main():
    print("== YOLO11n-Project: Main Workflow ==")
    # COCO class names
    coco_names = [
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
    # Colors for bounding boxes
    color_palette = [
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

    # 1. Load image
    orig_img, img_tensor, orig_shape, resized_shape = load_image(INPUT_IMAGE_PATH)
    print("Loaded input image.")

    # 2. Load PyTorch model and run inference
    model = load_pytorch_model(PT_MODEL_PATH)
    print("Loaded PyTorch model.")
    pt_boxes, pt_scores, pt_labels = run_pytorch_inference(model, img_tensor)
    print(f"PyTorch inference complete. {len(pt_boxes)} detections.")
    save_detection_image(
        orig_img,
        pt_boxes,
        pt_scores,
        pt_labels,
        OUTPUT_PT_IMAGE,
        class_names=coco_names,
        colors=color_palette,
    )
    print(f"Saved PyTorch result image to {OUTPUT_PT_IMAGE}")

    # 3. Convert to ONNX
    if not os.path.exists(ONNX_MODEL_PATH):
        convert_to_onnx(model, img_tensor, ONNX_MODEL_PATH)
        print(f"Converted model to ONNX: {ONNX_MODEL_PATH}")
    else:
        print(f"ONNX model already exists: {ONNX_MODEL_PATH}")

    # 4. Run ONNX inference
    # ONNX inference using Ultralytics pipeline (handles resizing and mapping)
    onnx_boxes, onnx_scores, onnx_labels = run_onnx_inference(
        ONNX_MODEL_PATH, INPUT_IMAGE_PATH
    )
    print(f"ONNX inference complete. {len(onnx_boxes)} detections.")

    save_detection_image(
        orig_img,
        onnx_boxes,
        onnx_scores,
        onnx_labels,
        OUTPUT_ONNX_IMAGE,
        class_names=coco_names,
        colors=color_palette,
    )
    print(f"Saved ONNX result image to {OUTPUT_ONNX_IMAGE}")

    # 5. IoU comparison
    iou_scores = compare_iou(pt_boxes, onnx_boxes)
    if len(iou_scores) > 0:
        plot_iou_chart(iou_scores, OUTPUT_IOU_CHART)
    print(f"Saved IoU chart to {OUTPUT_IOU_CHART}")

    # 6. Write summary (remove IoU section, since it's in iou_summary.txt)
    with open(OUTPUT_SUMMARY, "w") as f:
        f.write("YOLO11n-Project: Inference Summary\n\n")
        f.write("PyTorch Model Detections:\n")
        for i, (b, s, l) in enumerate(zip(pt_boxes, pt_scores, pt_labels)):
            f.write(f"  [{i}] Box: {b}, Score: {s:.2f}, Label: {coco_names[int(l)]}\n")
        f.write("\nONNX Model Detections:\n")
        for i, (b, s, l) in enumerate(zip(onnx_boxes, onnx_scores, onnx_labels)):
            f.write(f"  [{i}] Box: {b}, Score: {s:.2f}, Label: {coco_names[int(l)]}\n")

    with open(OUTPUT_IOU_SUMMARY, "w") as f:
        f.write("IoU Scores (PyTorch vs ONNX):\n")
        for i, score in enumerate(iou_scores):
            f.write(f"Detection {i}: IoU = {score:.3f}\n")
        f.write(f"\nMean IoU: {np.mean(iou_scores):.3f}\n")
    print(f"Saved summary to {OUTPUT_SUMMARY} and {OUTPUT_IOU_SUMMARY}")


if __name__ == "__main__":
    domain = DetectionDomain(
        INPUT_IMAGE_PATH,
        PT_MODEL_PATH,
        ONNX_MODEL_PATH,
        OUTPUT_PT_IMAGE,
        OUTPUT_ONNX_IMAGE,
        OUTPUT_SUMMARY,
        OUTPUT_IOU_CHART,
        OUTPUT_IOU_SUMMARY,
    )
    domain.run()
