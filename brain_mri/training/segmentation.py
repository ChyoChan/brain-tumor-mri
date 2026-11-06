"""Segmentation loss, training, validation, and evaluation metrics."""

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from brain_mri.utils import AverageMeter, EarlyStopping, fmt_time


class DiceBCELoss(nn.Module):
    """Combined Dice loss + Binary Cross-Entropy for binary segmentation.

    The two terms are weighted equally by default; adjust ``bce_weight``
    to shift the balance (e.g. 0.3 = mostly Dice-driven).
    """

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        pflat = probs.view(-1)
        tflat = targets.view(-1)
        intersection = (pflat * tflat).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            pflat.sum() + tflat.sum() + self.smooth
        )

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


def train_unet(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 15,
    save_path: str = "models/segmentation/best_unet.pth",
) -> Dict[str, list]:
    """Full U-Net training loop with validation, early stopping, and model saving.

    Returns a history dict with per-epoch train_loss, val_loss, val_dice,
    val_iou, val_hd95, and HD95 invalid-rate diagnostics.
    """
    model = model.to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )
    early_stop = EarlyStopping(patience=patience)

    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_hd95": [],
        "val_hd95_invalid_count": [],
        "val_hd95_invalid_rate": [],
        "lr": [],
    }
    best_dice = 0.0
    train_start = time.time()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}")
    print(f"  Training U-Net  |  {epochs} epochs  |  patience={patience}  |  device={device}")
    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")
    print(f"{'=' * 70}")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # ---- Train ----
        model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(train_loader, desc=f"  Train {epoch}/{epochs}", leave=False)
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), images.size(0))
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        train_loss = loss_meter.avg

        # ---- Validate ----
        val_loss, val_metrics = _validate_unet(model, val_loader, criterion, device)
        val_dice = val_metrics["dice"]
        val_iou = val_metrics["iou"]
        val_hd95 = val_metrics["hd95"]
        val_hd95_invalid_count = int(val_metrics.get("hd95_invalid_count", 0))
        val_hd95_invalid_rate = float(val_metrics.get("hd95_invalid_rate", 0.0))

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start
        eta = epoch_time * (epochs - epoch)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["val_hd95"].append(val_hd95)
        history["val_hd95_invalid_count"].append(val_hd95_invalid_count)
        history["val_hd95_invalid_rate"].append(val_hd95_invalid_rate)
        history["lr"].append(current_lr)

        improved = val_dice > best_dice
        marker = " *" if improved else ""
        hd95_suffix = f" ({val_hd95_invalid_count} invalid)" if val_hd95_invalid_count else ""

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"loss: {train_loss:.4f}/{val_loss:.4f} | "
            f"dice={val_dice:.4f} | "
            f"iou={val_iou:.4f} | "
            f"hd95={val_hd95:.4f}{hd95_suffix} | "
            f"lr={current_lr:.2e} | "
            f"{fmt_time(epoch_time)}/ep  ETA {fmt_time(eta)}{marker}"
        )

        if improved:
            best_dice = val_dice
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved (Dice={best_dice:.4f})")

        if early_stop(val_loss):
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    total_time = time.time() - train_start
    print(f"{'=' * 70}")
    print(f"  Training finished in {fmt_time(total_time)}  |  Best Dice={best_dice:.4f}")
    print(f"{'=' * 70}")

    return history


@torch.no_grad()
def _validate_unet(
    model: nn.Module,
    loader: DataLoader,
    criterion: DiceBCELoss,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Return (val_loss, mean_metrics) over the validation set."""
    model.eval()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    hd95_values: List[float] = []

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        loss_meter.update(loss.item(), images.size(0))

        preds = (torch.sigmoid(logits) > 0.5).float()
        for i in range(preds.size(0)):
            p = preds[i].cpu().numpy().squeeze().astype(np.uint8)
            g = masks[i].cpu().numpy().squeeze().astype(np.uint8)
            sample_metrics = compute_segmentation_metrics(p, g)
            dice_meter.update(sample_metrics["dice"])
            iou_meter.update(sample_metrics["iou"])
            hd95_values.append(
                float(sample_metrics.get("hd95", sample_metrics.get("hausdorff95", float("inf"))))
            )

    hd95_array = np.asarray(hd95_values, dtype=np.float64)
    finite_hd95 = hd95_array[np.isfinite(hd95_array)]
    hd95_invalid_count = int((~np.isfinite(hd95_array)).sum())
    hd95_mean = float(np.mean(finite_hd95)) if finite_hd95.size else float("inf")

    metrics = {
        "dice": float(dice_meter.avg),
        "iou": float(iou_meter.avg),
        "hd95": hd95_mean,
        "hd95_invalid_count": hd95_invalid_count,
        "hd95_invalid_rate": float(hd95_invalid_count / hd95_array.size) if hd95_array.size else 0.0,
    }
    return loss_meter.avg, metrics


def compute_segmentation_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    voxel_spacing: Tuple[float, ...] = (1.0, 1.0),
) -> Dict[str, float]:
    """Compute Dice, IoU, HD95, sensitivity, specificity for binary masks."""
    from medpy.metric.binary import dc as dice_coefficient
    from medpy.metric.binary import hd95 as hausdorff_distance_95

    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)
    pred_fg = pred_bin.astype(bool)
    gt_fg = gt_bin.astype(bool)

    if gt_bin.sum() == 0 and pred_bin.sum() == 0:
        return {
            "dice": 1.0,
            "iou": 1.0,
            "hausdorff95": 0.0,
            "hd95": 0.0,
            "sensitivity": 1.0,
            "specificity": 1.0,
        }

    if gt_bin.sum() == 0 or pred_bin.sum() == 0:
        return {
            "dice": 0.0,
            "iou": 0.0,
            "hausdorff95": np.inf,
            "hd95": np.inf,
            "sensitivity": 0.0,
            "specificity": 0.0,
        }

    dsc = dice_coefficient(pred_bin, gt_bin)
    intersection = np.logical_and(pred_fg, gt_fg).sum()
    union = np.logical_or(pred_fg, gt_fg).sum()
    iou = intersection / (union + 1e-8)

    try:
        hd = hausdorff_distance_95(pred_bin, gt_bin, voxelspacing=voxel_spacing)
    except Exception:
        hd = float("inf")

    tp = np.logical_and(pred_fg, gt_fg).sum()
    fn = np.logical_and(~pred_fg, gt_fg).sum()
    fp = np.logical_and(pred_fg, ~gt_fg).sum()
    tn = np.logical_and(~pred_fg, ~gt_fg).sum()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    return {
        "dice": float(dsc),
        "iou": float(iou),
        "hausdorff95": float(hd),
        "hd95": float(hd),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def evaluate_segmentation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Evaluate a U-Net model on a DataLoader. Returns (avg_metrics, std_metrics)."""
    model.eval()
    all_metrics: List[Dict[str, float]] = []

    for images, masks in tqdm(loader, desc="Evaluating segmentation", leave=False):
        images = images.to(device)
        with torch.no_grad():
            logits = model(images)
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        gt = masks.cpu().numpy()

        for i in range(preds.shape[0]):
            m = compute_segmentation_metrics(preds[i].squeeze(), gt[i].squeeze())
            all_metrics.append(m)

    if not all_metrics:
        empty = {
            "dice": 0.0,
            "iou": 0.0,
            "hausdorff95": 0.0,
            "hd95": 0.0,
            "hd95_invalid_count": 0.0,
            "hd95_invalid_rate": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
        }
        return empty, empty

    avg = {
        k: float(np.mean([m[k] for m in all_metrics]))
        for k in all_metrics[0]
        if k not in {"hd95", "hausdorff95"}
    }
    std = {
        k: float(np.std([m[k] for m in all_metrics]))
        for k in all_metrics[0]
        if k not in {"hd95", "hausdorff95"}
    }

    hd95_values = np.asarray(
        [m.get("hd95", m.get("hausdorff95", float("inf"))) for m in all_metrics],
        dtype=np.float64,
    )
    finite_hd95 = hd95_values[np.isfinite(hd95_values)]
    hd95_invalid_count = int((~np.isfinite(hd95_values)).sum())

    hd95_avg = float(np.mean(finite_hd95)) if finite_hd95.size else float("inf")
    hd95_std = float(np.std(finite_hd95)) if finite_hd95.size else float("inf")

    avg["hd95"] = hd95_avg
    avg["hausdorff95"] = hd95_avg
    avg["hd95_invalid_count"] = float(hd95_invalid_count)
    avg["hd95_invalid_rate"] = float(hd95_invalid_count / hd95_values.size) if hd95_values.size else 0.0

    std["hd95"] = hd95_std
    std["hausdorff95"] = hd95_std
    std["hd95_invalid_count"] = 0.0
    std["hd95_invalid_rate"] = 0.0

    return avg, std
