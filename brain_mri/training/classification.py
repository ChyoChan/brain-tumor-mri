"""Classification training, evaluation, and hyperparameter search."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

from brain_mri.constants import CLASS_NAMES
from brain_mri.checkpoints import save_metadata
from brain_mri.data.classification import (
    BrainTumorDataset,
    get_train_transforms,
    get_val_transforms,
)
from brain_mri.models.classification import build_resnet50
from brain_mri.utils import AverageMeter, EarlyStopping, fmt_time


def compute_class_weights(labels: List[int], num_classes: int = 4) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = 1.0 / (counts + 1e-6)
    weights /= weights.sum()
    weights *= num_classes
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_str: str = "",
) -> float:
    model.train()
    loss_meter = AverageMeter()
    correct, total = 0, 0

    pbar = tqdm(loader, desc=f"  Train {epoch_str}", leave=False, unit="batch")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{correct/total:.3f}")

    return loss_meter.avg


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "  Val",
) -> Tuple[float, Dict[str, float], np.ndarray]:
    """Return (val_loss, metrics_dict, confusion_matrix)."""
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=desc, leave=False, unit="batch")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_meter.update(loss.item(), images.size(0))
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES))))
    return loss_meter.avg, metrics, cm


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    epochs: int = 50,
    patience: int = 10,
    save_path: str = "models/classification/best_resnet50.pth",
    checkpoint_metadata: Optional[dict] = None,
) -> Dict[str, list]:
    """Full training loop with early stopping. Returns history dict."""
    early_stop = EarlyStopping(patience=patience)
    history: Dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "accuracy": [], "precision": [], "recall": [], "f1": [],
        "lr": [], "epoch_time": [],
    }
    best_f1 = -float("inf")
    train_start = time.time()

    print(f"{'='*70}")
    print(f"  Training ResNet50  |  {epochs} epochs  |  patience={patience}  |  device={device}")
    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")
    print(f"{'='*70}")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        epoch_str = f"{epoch}/{epochs}"

        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch_str=epoch_str,
        )
        val_loss, metrics, _ = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start
        eta = epoch_time * (epochs - epoch)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        history["epoch_time"].append(epoch_time)
        for k in ("accuracy", "precision", "recall", "f1"):
            history[k].append(metrics[k])

        improved = metrics["f1"] > best_f1
        marker = " *" if improved else ""

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"loss: {train_loss:.4f}/{val_loss:.4f} | "
            f"acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f} | "
            f"lr={current_lr:.2e} | "
            f"{fmt_time(epoch_time)}/ep  ETA {fmt_time(eta)}{marker}"
        )

        if improved:
            best_f1 = metrics["f1"]
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            if checkpoint_metadata is not None:
                save_metadata(save_path, checkpoint_metadata)
            print(f"  -> Best model saved (F1={best_f1:.4f})")

        if early_stop(val_loss):
            print(f"\nEarly stopping at epoch {epoch} (no val_loss improvement for {patience} epochs)")
            break

    total_time = time.time() - train_start
    print(f"{'='*70}")
    print(f"  Training finished in {fmt_time(total_time)}  |  Best F1={best_f1:.4f}")
    print(f"{'='*70}")

    return history


def grid_search_hyperparams(
    train_samples: List[Tuple[str, int]],
    val_samples: List[Tuple[str, int]],
    lr_candidates: List[float],
    bs_candidates: List[int],
    device: torch.device,
    epochs: int = 15,
    num_classes: int = 4,
    input_size: int = 224,
    aug_cfg: Optional[dict] = None,
) -> Dict:
    """Run a lightweight grid search over LR × batch_size."""
    results = []
    best_f1 = 0.0
    best_params = {}

    labels = [s[1] for s in train_samples]
    class_weights = compute_class_weights(labels, num_classes).to(device)

    combos = [(lr, bs) for lr in lr_candidates for bs in bs_candidates]
    for combo_i, (lr, bs) in enumerate(combos, 1):
        print(f"\n[{combo_i}/{len(combos)}] Grid search: lr={lr}, bs={bs}")
        t0 = time.time()
        train_ds = BrainTumorDataset(train_samples, get_train_transforms(input_size, aug_cfg))
        val_ds = BrainTumorDataset(val_samples, get_val_transforms(input_size))
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2)

        model = build_resnet50(num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        for epoch in tqdm(range(1, epochs + 1), desc=f"  lr={lr} bs={bs}", leave=False):
            train_one_epoch(model, train_loader, criterion, optimizer, device)

        _, metrics, _ = evaluate(model, val_loader, criterion, device, desc="  Eval")
        f1 = metrics["f1"]
        elapsed = time.time() - t0
        print(f"  Result: acc={metrics['accuracy']:.4f}, f1={f1:.4f} ({fmt_time(elapsed)})")
        results.append({"lr": lr, "bs": bs, **metrics})

        if f1 > best_f1:
            best_f1 = f1
            best_params = {"lr": lr, "bs": bs}

    print(f"\nBest grid-search params: {best_params} (F1={best_f1:.4f})")
    return {"results": results, "best_params": best_params, "best_f1": best_f1}
