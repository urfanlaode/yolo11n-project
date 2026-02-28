import glob
import os

import cv2
import torch

# Dynamically find the first image file named 'image.*' in the inputs directory
image_candidates = glob.glob(os.path.join("inputs", "image.*"))
INPUT_IMAGE_PATH = image_candidates[0] if image_candidates else None


def load_image(image_path, img_size=640):
    """
    Load and preprocess an image for model inference.
    Supports PNG, JPG, JPEG, WEBP, and other formats supported by OpenCV.

    Returns:
        orig_img: Original BGR image.
        img_tensor: Preprocessed image tensor [1, 3, H, W].
        orig_shape: Original image shape.
        resized_shape: Resized image shape.
    """
    # Try reading with OpenCV (supports png, jpg, jpeg, webp, bmp, tiff, etc.)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found or unsupported format: {image_path}")
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
        # Make font size proportional to image height (or width), but smaller
        font_scale = (
            max(img.shape[0], img.shape[1]) / 1600.0
        )  # Increased divisor for smaller font
        font_thickness = max(1, int(font_scale * 2))
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        margin_x, margin_y = 2, 2
        text_x = x1 + margin_x
        text_y = max(0, y1 - text_height - margin_y)
        # Draw label text only, no background, using border color
        cv2.putText(
            img,
            label_text,
            (
                text_x,
                text_y + text_height,
            ),  # OpenCV puts text baseline at y, so add text_height
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,  # Use border color for text
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    cv2.imwrite(out_path, img)
