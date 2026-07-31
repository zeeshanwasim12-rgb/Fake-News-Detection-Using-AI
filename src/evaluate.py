"""
Week 4 - Evaluation
--------------------
Accuracy, precision, recall, F1-score and confusion matrix computed by
hand (no sklearn.metrics), plus simple matplotlib visualizations.
"""

import numpy as np


def confusion_matrix(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return np.array([[tn, fp], [fn, tp]]), (tp, tn, fp, fn)


def classification_metrics(y_true, y_pred):
    cm, (tp, tn, fp, fn) = confusion_matrix(y_true, y_pred)
    accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


def print_report(name, metrics):
    print(f"\n=== {name} ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print("Confusion Matrix [[TN FP] [FN TP]]:")
    print(metrics["confusion_matrix"])


def plot_results(results: dict, save_path: str):
    """results: {model_name: metrics_dict}. Saves a bar chart comparing
    accuracy/precision/recall/f1 across models plus confusion matrices."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results.keys())
    metrics_names = ["accuracy", "precision", "recall", "f1"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart comparison
    x = np.arange(len(names))
    width = 0.2
    for i, m in enumerate(metrics_names):
        vals = [results[n][m] for n in names]
        axes[0].bar(x + i * width, vals, width, label=m)
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(names, rotation=20)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Model comparison")
    axes[0].legend()

    # Confusion matrix of best model (by f1)
    best_name = max(names, key=lambda n: results[n]["f1"])
    cm = results[best_name]["confusion_matrix"]
    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_title(f"Confusion Matrix - {best_name} (best F1)")
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["Pred Real", "Pred Fake"])
    axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(["True Real", "True Fake"])
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(results: dict, save_path: str):
    """Save confusion matrices for every evaluated model in one figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4))
    axes = np.atleast_1d(axes)
    for axis, name in zip(axes, names):
        cm = results[name]["confusion_matrix"]
        image = axis.imshow(cm, cmap="Blues")
        axis.set_title(name)
        axis.set_xticks([0, 1], ["Pred Real", "Pred Fake"])
        axis.set_yticks([0, 1], ["True Real", "True Fake"])
        for i in range(2):
            for j in range(2):
                axis.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=axes.tolist(), shrink=0.8)
    fig.suptitle("Confusion Matrices by Model", fontsize=14, fontweight="bold")
    fig.subplots_adjust(top=0.78, wspace=0.45)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
