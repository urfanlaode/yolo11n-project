import os

import cv2
import torch

INPUT_IMAGE_PATH = os.path.join("inputs", "image.png")


def load_image(image_path, img_size=640):
    """
    Load and preprocess an image for model inference.

    Returns:
        orig_img: Original BGR image.
        img_tensor: Preprocessed image tensor [1, 3, H, W].
        orig_shape: Original image shape.
        resized_shape: Resized image shape.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    return (
        img,
        img_tensor.unsqueeze(0),
        img.shape[:2],
        img_resized.shape[:2],
    )


def save_detection_image(
    orig_img, boxes, scores, labels, out_path, class_names=None, colors=None
):
    """
    Draw bounding boxes and labels on an image and save to disk.

    Args:
        orig_img: Original image (BGR).
        boxes: List of bounding boxes.
        scores: List of confidence scores.
        labels: List of class labels.
        out_path: Output file path.
        class_names: Optional list of class names.
        colors: Optional list of colors for boxes.
    """
    img = orig_img.copy()
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = map(int, box)
        color = colors[i % len(colors)] if colors else (0, 255, 0)
        label_text = (
            f"{class_names[int(label)]}:{score:.2f}"
            if class_names is not None
            else f"{int(label)}:{score:.2f}"
        )
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        font_scale = 0.4
        font_thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        margin_x, margin_y = 2, 2
        text_x = x1 + margin_x
        text_y = y1 + text_height + margin_y
        cv2.rectangle(
            img,
            (text_x - 1, text_y - text_height - 1),
            (text_x + text_width + 1, text_y + baseline + 1),
            color,
            -1,
        )
        text_color = (255, 255, 255) if sum(color) < 384 else (0, 0, 0)
        cv2.putText(
            img,
            label_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    cv2.imwrite(out_path, img)
