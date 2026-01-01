import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

from PIL import Image


def plot_train_and_validation_error(epochs, training_loss, validation_loss, figsize=(10, 4)):
    if len(epochs) == 0:
        print("No metrics to plot.")
    else:
        plt.figure(figsize=figsize)
        plt.plot(epochs, training_loss, marker='o', label="Training Loss")
        plt.plot(epochs, validation_loss, marker='s', label="Validation Loss")

        plt.title("Training and Validation per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.show()


def plot_dice_and_iou(epochs, dice_scores, iou_scores, figsize=(10, 4)):
    if len(epochs) == 0:
        print("No metrics to plot.")
    else:
        plt.figure(figsize=figsize)
        plt.plot(epochs, dice_scores, marker='o', label="Dice Score")
        plt.plot(epochs, iou_scores, marker='s', label="IoU Score")

        plt.title("Dice and IoU per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.ylim(0, 1.05)  # Metrics range from 0 to 1
        plt.grid(True)
        plt.legend()
        plt.show()


def plot_confusion_matrices2(epochs, cms, figsize_per_matrix=(3, 3)):
    """
        Plot confusion matrices in a grid (4 per row) and show Accuracy, Precision, Recall, F1 below each matrix.
        """
    num_cms = len(cms)
    cols = 4
    rows = math.ceil(num_cms / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per_matrix[0] * cols, figsize_per_matrix[1] * rows))
    axes = axes.flatten()

    for i, cm in enumerate(cms):
        TP, TN, FP, FN = cm["TP"], cm["TN"], cm["FP"], cm["FN"]

        # Compute metrics
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        matrix = [[TP, FP],
                  [FN, TN]]
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[i])

        axes[i].set_title(f"Epoch {epochs[i]}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

        # Add metrics as text below the matrix
        metrics_text = f"Acc={accuracy:.3f}  P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}"
        axes[i].text(0.5, -0.3, metrics_text, ha='center', va='top', transform=axes[i].transAxes, fontsize=8)

    # Hide unused subplots
    for j in range(num_cms, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def plot_all_roc_curves(epochs, fprs, tprs, aucs, figsize=(10, 5)):
    """
    Plot all ROC curves in one graph for multiple epochs.
    """
    plt.figure(figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))

    for ep, fpr, tpr, auc_val, color in zip(epochs, fprs, tprs, aucs, colors):
        if len(fpr) == 0 or len(tpr) == 0:
            continue  # Skip missing data
        plt.plot(fpr, tpr, color=color, label=f"Epoch {ep} (AUC={auc_val:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves per Epoch")
    ncol = 2 if len(epochs) <= 10 else 3
    plt.legend(loc="lower right", fontsize="small", ncol=ncol)
    plt.grid(True)
    plt.show()


def show_predicted_images(image_folder):
    # Get list of images in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Take first 3 images
    image_files = image_files[:3]

    # Display images stacked vertically (1 column, 3 rows)
    fig, axes = plt.subplots(len(image_files), 1, figsize=(10, 7))
    for ax, img_file in zip(axes, image_files):
        img = Image.open(os.path.join(image_folder, img_file))
        ax.imshow(img)
        ax.set_title(img_file)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
