import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from src.util import (
    dice_metric, iou_metric,
    sigmoid, confusion_matrix_pixelwise, dice_loss, compute_roc_auc_metrics,
)
from src.save_data import save_combined_images


def train(loader, model, optimizer, bce_loss_fun, device='cpu', clip_grad=None):
    model.train()
    total_loss = 0

    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data, targets = data.to(device), targets.to(device)
        targets = targets.float()

        # forward
        predictions = model(data)
        targets = model.formate(targets, predictions)
        loss = bce_loss_fun(predictions, targets) + dice_loss(predictions, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        # update tqdm loop
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    print(f"Training loss: {avg_loss:.4f}")
    return avg_loss


def validate(loader, model, bce_loss_fun, device="cpu", threshold=0.5):
    model.eval()

    total_loss = 0
    dice_total = 0
    iou_total = 0
    confusion = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    all_probs = []
    all_targets = []

    with torch.no_grad():
        loop = tqdm(loader)
        for data, targets in loop:
            data, targets = data.to(device), targets.to(device)
            targets = targets.float()

            preds = model(data)
            targets = model.formate(targets, preds)

            loss = bce_loss_fun(preds, targets) + dice_loss(preds, targets)
            total_loss += loss.item()

            probs, preds_bin = sigmoid(preds, threshold=threshold)

            dice_total += dice_metric(targets, preds_bin)
            iou_total += iou_metric(targets, preds_bin)

            # Confusion
            cm_batch = confusion_matrix_pixelwise(targets, preds, threshold)
            for k in confusion.keys():
                confusion[k] += cm_batch[k]

                # Store probabilities for ROC/AUC
                all_probs.append(probs.cpu())
                all_targets.append(targets.cpu())

                # # Save probabilities for ROC/AUC: ensure targets are strictly 0/1
                # all_probs.append(probs.cpu().view(-1))
                # all_targets.append((targets.cpu().view(-1) > 0.5).int())

    # Average Dice and IoU
    dice_score = dice_total / len(loader)
    iou_score = iou_total / len(loader)
    avg_loss = total_loss / len(loader)

    print(f"Dice metric: {dice_score:.4f}")
    print(f"Iou metric: {iou_score:.4f}")
    print(f"Validate loss: {avg_loss:.4f}")

    # Concatenate all batches
    all_probs = torch.cat(all_probs, dim=0).view(-1).numpy()
    all_targets = torch.cat(all_targets, dim=0).view(-1).numpy()

    # Compute ROC and AUC
    fpr, tpr, thresholds, roc_auc = compute_roc_auc_metrics(all_targets, all_probs)

    return avg_loss, dice_score, iou_score, confusion, fpr, tpr, roc_auc


def test(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    os.makedirs(folder, exist_ok=True)

    for idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        y = y.float()

        with torch.no_grad():
            # Forward
            out = model(x)
            _, preds = sigmoid(out)

        for i in range(x.shape[0]):
            # Original image
            img = x[i].cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            img = np.clip(img, 0, 1) * 255
            img = img.astype(np.uint8)

            if img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            H_img, W_img, _ = img.shape

            # Ground truth mask
            mask = y[i].cpu().numpy().squeeze()
            mask = (mask * 255).astype(np.uint8)
            mask = cv2.resize(mask, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
            mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # Predicted mask (ZERO PAD ONLY)
            pred_mask = preds[i].cpu().numpy().squeeze()
            H_pred, W_pred = pred_mask.shape
            pad_top = (H_img - H_pred) // 2
            pad_bottom = H_img - H_pred - pad_top
            pad_left = (W_img - W_pred) // 2
            pad_right = W_img - W_pred - pad_left

            pred_mask = np.pad(
                pred_mask,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=0
            )
            pred_mask = (pred_mask * 255).astype(np.uint8)
            pred_mask_3c = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)

            # Overlay
            overlay = img.copy()
            overlay[pred_mask > 128] = (0, 255, 0)
            overlay = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

            save_combined_images(img, mask_3c, pred_mask_3c, overlay, idx, i, folder)

        print(f"Saved batch {idx} paper-style visualizations")
