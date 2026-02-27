import numpy as np


def iou(boxA, boxB):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def compare_iou(pt_boxes, onnx_boxes, threshold=0.5):
    """Return best IoU scores for each box in pt_boxes against onnx_boxes."""
    iou_scores = []
    for boxA in pt_boxes:
        best_iou = 0
        for boxB in onnx_boxes:
            score = iou(boxA, boxB)
            if score > best_iou:
                best_iou = score
        iou_scores.append(best_iou)
    return iou_scores


def compare_iou_matrix(boxesA, boxesB):
    """Return IoU matrix between two sets of boxes."""
    matrix = np.zeros((len(boxesA), len(boxesB)), dtype=np.float32)
    for i, boxA in enumerate(boxesA):
        for j, boxB in enumerate(boxesB):
            matrix[i, j] = iou(boxA, boxB)
    return matrix
