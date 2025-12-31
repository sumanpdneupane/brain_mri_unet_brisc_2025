import torch
from sklearn.metrics import roc_curve, auc
import numpy as np


def crop_tensor(enc_feat, dec_feat):
    """
    Crop encoder feature to match decoder feature shape.
    (center crop)
    """
    _, _, H, W = dec_feat.size()
    _, _, H_enc, W_enc = enc_feat.size()

    crop_top = (H_enc - H) // 2
    crop_left = (W_enc - W) // 2

    return enc_feat[:, :, crop_top: crop_top + H, crop_left: crop_left + W]


def softmax(x):
    # Multi-class segmentation: for multi-class: pixel-wise softmax
    preds_prob = torch.softmax(x, dim=1)  # softmax across channels
    preds_bin = torch.argmax(x, dim=1, keepdim=True).float()
    return preds_prob, preds_bin


def sigmoid(x, threshold=0.5):
    # Binary segmentation: for binary: pixel-wise softmax
    preds_prob = torch.sigmoid(x)
    preds_bin = (preds_prob > threshold).float()
    return preds_prob, preds_bin


# Dice for training (differentiable)
def dice_loss(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)  # convert logits to probabilities
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice_score = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice_score.mean()  # return as loss


def dice_metric(y, preds, smooth=1e-8):
    """
        y, preds: [B, 1, H, W] with values 0 or 1
        returns: mean Dice over batch
        """
    y = y.reshape(y.size(0), -1)
    preds = preds.reshape(preds.size(0), -1)

    intersection = (preds * y).sum(dim=1)
    dice = (2 * intersection + smooth) / (preds.sum(dim=1) + y.sum(dim=1) + smooth)
    return dice.mean().item()


# IoU for metrics (evaluation, not training)
def iou_metric(y, preds, smooth=1e-8):
    """
        y, preds: [B, 1, H, W] binary masks
        returns: mean IoU over batch
        """
    y = y.reshape(y.size(0), -1)
    preds = preds.reshape(preds.size(0), -1)

    intersection = (preds * y).sum(dim=1)
    union = preds.sum(dim=1) + y.sum(dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def confusion_matrix_pixelwise(y_true, y_pred, threshold=0.5):
    """
    Compute per-pixel confusion matrix for binary segmentation.
    """
    y_pred_bin = (y_pred > threshold).float()

    # Flatten batch
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred_bin.reshape(-1)

    TP = ((y_true_f == 1) & (y_pred_f == 1)).sum().item()
    TN = ((y_true_f == 0) & (y_pred_f == 0)).sum().item()
    FP = ((y_true_f == 0) & (y_pred_f == 1)).sum().item()
    FN = ((y_true_f == 1) & (y_pred_f == 0)).sum().item()

    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}


def compute_roc_auc_metrics(y_true, y_probs, threshold=0.5):
    """
    Compute ROC curve metrics and AUC for binary segmentation without plotting.
    """
    # Convert tensors to numpy if needed
    if not isinstance(y_true, np.ndarray):
        y_true = y_true.reshape(-1).cpu().numpy()
    if not isinstance(y_probs, np.ndarray):
        y_probs = y_probs.reshape(-1).cpu().numpy()

    # Ensure y_true is strictly binary 0 or 1
    y_true = (y_true > threshold).astype(int)

    fpr, tpr, thres = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, thres, roc_auc
