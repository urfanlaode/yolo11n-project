from ultralytics import YOLO


def load_pytorch_model(model_path):
    """Load a PyTorch YOLO model from the given path."""
    return YOLO(model_path)


def run_pytorch_inference(model, img_tensor, image_path=None):
    """Run inference using a PyTorch YOLO model."""
    results = model(image_path) if image_path is not None else model(img_tensor)
    r = results[0]
    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    labels = r.boxes.cls.cpu().numpy()
    return boxes, scores, labels


def convert_to_onnx(model, img_tensor, onnx_path):
    """Export a PyTorch YOLO model to ONNX format."""
    model.export(format="onnx", optimize=True, imgsz=img_tensor.shape[-1], device="cpu")


def run_onnx_inference(onnx_path, image_path, conf_thres=0.25, iou_thres=0.45):
    """Run inference using an ONNX YOLO model via Ultralytics pipeline."""
    model = YOLO(onnx_path)
    results = model.predict(image_path)
    r = results[0]
    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    labels = r.boxes.cls.cpu().numpy()
    return boxes, scores, labels
