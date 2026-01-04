import csv
import os
import cv2
import numpy as np
import torch


def save_checkpoint(epoch, model, optimizer, foldername="checkpoints", filename="model_checkpoint.pth.tar"):
    checkpoint_dir = os.path.join(".", foldername)
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)

    checkpoint = {
        'epoch': epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(model, optimizer, foldername="checkpoints", filename="model_checkpoint.pth.tar"):
    checkpoint_dir = os.path.join(".", foldername)
    filepath = os.path.join(checkpoint_dir, filename)
    if not os.path.exists(filepath):
        print(f"Checkpoint '{filepath}' not found. Skipping load.")
        return

    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint['optimizer'])
    last_epoch = checkpoint.get("epoch", 1)

    print(f"Model loaded from {filepath} (last epoch: {last_epoch})")
    return model, optimizer, checkpoint['epoch']


def save_combined_images(original_image, original_mask, pred_mask, overlay, idx, i, folder="saved_images/"):
    H_img, W_img, _ = original_image.shape

    # Title settings
    titles = [
        "Original Image",
        "Ground Truth Mask",
        "Predicted Mask",
        "Overlay"
    ]

    title_height = 50
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    font_color = (255, 255, 255)
    bg_color = (40, 40, 40)
    separator_width = 5

    # Vertical separators
    sep = np.ones((H_img, separator_width, 3), dtype=np.uint8) * 255
    combined_row = np.concatenate(
        [original_image, sep, original_mask, sep, pred_mask, sep, overlay], axis=1
    )

    # Title row
    combined_width = combined_row.shape[1]
    title_row = np.zeros((title_height, combined_width, 3), dtype=np.uint8)
    title_row[:] = bg_color
    panel_width = W_img + separator_width
    panel_starts = [
        0,
        panel_width,
        2 * panel_width,
        3 * panel_width
    ]

    for j, title in enumerate(titles):
        text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
        text_x = panel_starts[j] + (W_img - text_size[0]) // 2
        text_y = (title_height + text_size[1]) // 2

        cv2.putText(title_row, title, (text_x, text_y),
                    font, font_scale, font_color, thickness, cv2.LINE_AA
                    )

    # Stack title + images
    final_img = np.concatenate([title_row, combined_row], axis=0)

    # Save
    cv2.imwrite(
        os.path.join(folder, f"combine_image_{idx}_{i}.png"),
        cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
    )

def save_metrics_summary(
        epoch, training_loss, validation_loss,
        dice_score, iou_score,
        folder="metrics", filename="metrics_summary.csv"
):
    # Save Dice, IoU, and AUC into a CSV file (one row per epoch).
    # Create folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Full path to CSV
    summary_csv = os.path.join(folder, filename)

    # Check if CSV already exists
    summary_exists = os.path.exists(summary_csv)
    with open(summary_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not summary_exists:
            writer.writerow(["Epoch", "Training Loss", "Validation Loss", "Dice", "IoU"])
        writer.writerow([epoch, training_loss, validation_loss, dice_score, iou_score])

def load_metrics_summary(folder="metrics", filename="metrics_summary.csv"):
    # Load Dice, IoU, and AUC metrics per epoch from CSV.
    summary_csv = os.path.join(folder, filename)
    epochs, training_losses, validation_losses, dice_scores, iou_scores = [], [], [], [], []

    if not os.path.exists(summary_csv):
        print(f"No metrics summary found at {summary_csv}")
        return epochs, training_losses, validation_losses, dice_scores, iou_scores

    with open(summary_csv, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["Epoch"]))
            training_losses.append(float(row["Training Loss"]))
            validation_losses.append(float(row["Validation Loss"]))
            dice_scores.append(float(row["Dice"]))
            iou_scores.append(float(row["IoU"]))

    return epochs, training_losses, validation_losses, dice_scores, iou_scores


def save_confusion_matrix(epoch, cm, folder="metrics", filename="confusion_matrix.csv"):
    """
    cm: dictionary {"TP":..., "TN":..., "FP":..., "FN":...}
    """
    os.makedirs(folder, exist_ok=True)
    summary_csv = os.path.join(folder, filename)
    summary_exists = os.path.exists(summary_csv)

    with open(summary_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not summary_exists:
            writer.writerow(["Epoch", "TP", "TN", "FP", "FN"])
        writer.writerow([epoch, cm.get("TP",0), cm.get("TN",0), cm.get("FP",0), cm.get("FN",0)])

def load_confusion_matrices(folder="metrics", filename="confusion_matrix.csv"):
    summary_csv = os.path.join(folder, filename)
    confusion_matrices = []

    if not os.path.exists(summary_csv):
        print(f"No confusion matrix summary found at {summary_csv}")
        return confusion_matrices

    with open(summary_csv, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cm = {
                "TP": int(row.get("TP", 0)),
                "TN": int(row.get("TN", 0)),
                "FP": int(row.get("FP", 0)),
                "FN": int(row.get("FN", 0)),
            }
            confusion_matrices.append(cm)

    return confusion_matrices



def save_roc_auc_summary(epoch, fpr, tpr, roc_auc, folder="metrics/roc_auc"):
    """
    Save ROC curves (fpr, tpr) as .npy and AUC as CSV for each epoch.
    """
    os.makedirs(folder, exist_ok=True)

    # Save FPR and TPR arrays
    np.save(os.path.join(folder, f"roc_fpr_epoch{epoch}.npy"), fpr)
    np.save(os.path.join(folder, f"roc_tpr_epoch{epoch}.npy"), tpr)

    # Save scalar AUC in CSV
    auc_file = os.path.join(folder, "roc_auc.csv")
    write_header = not os.path.exists(auc_file)
    with open(auc_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Epoch", "AUC"])
        writer.writerow([epoch, roc_auc])

def load_roc_auc_summary(num_epochs, folder="metrics/roc_auc"):
    """
    Load all ROC curves and AUC values for plotting.
    """
    fprs, tprs, aucs = [], [], []

    for epoch in range(num_epochs):
        fpr_path = os.path.join(folder, f"roc_fpr_epoch{epoch}.npy")
        tpr_path = os.path.join(folder, f"roc_tpr_epoch{epoch}.npy")

        if os.path.exists(fpr_path) and os.path.exists(tpr_path):
            fprs.append(np.load(fpr_path))
            tprs.append(np.load(tpr_path))
        else:
            print(f"Warning: ROC data missing for epoch {epoch}")
            fprs.append(np.array([]))
            tprs.append(np.array([]))

    # Load AUC values
    auc_file = os.path.join(folder, "roc_auc.csv")
    if os.path.exists(auc_file):
        with open(auc_file, "r") as f:
            reader = csv.DictReader(f)
            # Assumes CSV has rows in epoch order
            for row in reader:
                aucs.append(float(row["AUC"]))
    else:
        print("Warning: AUC CSV not found.")

    return fprs, tprs, aucs
