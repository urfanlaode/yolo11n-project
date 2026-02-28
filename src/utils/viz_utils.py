import matplotlib.pyplot as plt


def plot_iou_chart(iou_scores, out_path, pt_scores=None, onnx_scores=None):
    """
    Create a grouped bar chart comparing PyTorch and ONNX detection scores, with IoU overlay.

    Args:
        iou_scores (list of float): IoU scores for each detection.
        out_path (str): Output path for the chart image.
        pt_scores (list of float, optional): PyTorch detection scores.
        onnx_scores (list of float, optional): ONNX detection scores.
    """
    plt.figure(figsize=(10, 5))
    # Determine the minimum length among all lists
    n_iou = len(iou_scores)
    n_pt = len(pt_scores) if pt_scores is not None else n_iou
    n_onnx = len(onnx_scores) if onnx_scores is not None else n_iou
    n = min(n_iou, n_pt, n_onnx)
    indices = range(n)
    bar_width = 0.35

    # Pad or trim scores to match n
    def pad_or_trim(arr, n):
        arr = arr if arr is not None else [0] * n
        if len(arr) < n:
            arr = list(arr) + [0] * (n - len(arr))
        return arr[:n]

    iou_scores_plot = pad_or_trim(iou_scores, n)
    pt_scores_plot = pad_or_trim(pt_scores, n) if pt_scores is not None else None
    onnx_scores_plot = pad_or_trim(onnx_scores, n) if onnx_scores is not None else None

    if pt_scores_plot is not None and onnx_scores_plot is not None:
        plt.bar(
            [i - bar_width / 2 for i in indices],
            pt_scores_plot,
            width=bar_width,
            label="PyTorch Score",
            color="#1f77b4",
            alpha=0.7,
        )
        plt.bar(
            [i + bar_width / 2 for i in indices],
            onnx_scores_plot,
            width=bar_width,
            label="ONNX Score",
            color="#ff7f0e",
            alpha=0.7,
        )
        plt.plot(
            indices, iou_scores_plot, "ko-", label="IoU (pt vs onnx)", markersize=6
        )
        plt.ylabel("Score / IoU")
    else:
        plt.bar(
            indices,
            iou_scores_plot,
            width=bar_width,
            label="IoU (pt vs onnx)",
            color="#1f77b4",
            alpha=0.7,
        )
        plt.ylabel("IoU")

    plt.xlabel("Detection Index")
    plt.title("Detection Scores and IoU: PyTorch vs ONNX")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
