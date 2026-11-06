"""Classification datasets and image transforms."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def get_train_transforms(input_size: int = 224, cfg: Optional[dict] = None) -> transforms.Compose:
    cfg = cfg or {}
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomRotation(cfg.get("rotation_degrees", 15)),
        transforms.RandomHorizontalFlip(0.5 if cfg.get("horizontal_flip", True) else 0.0),
        transforms.ColorJitter(
            brightness=cfg.get("brightness", 0.2),
            contrast=cfg.get("contrast", 0.2),
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_val_transforms(input_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class BrainTumorDataset(Dataset):
    """Dataset that reads (image_path, label) pairs.

    Non-existent paths are filtered out at construction time so that
    DataLoader workers (num_workers > 0) never hit FileNotFoundError.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform: Optional[transforms.Compose] = None,
        grayscale: bool = False,
    ):
        valid = [(p, l) for p, l in samples if Path(p).exists()]
        if len(valid) < len(samples):
            import warnings
            warnings.warn(
                f"BrainTumorDataset: dropped {len(samples) - len(valid)} "
                f"missing files ({len(valid)} remaining)"
            )
        self.samples = valid
        self.transform = transform
        self.grayscale = grayscale

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.grayscale:
            img = img.convert("L").convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_original_lookup(samples: List[Tuple[str, int]]) -> Dict[Tuple[int, str], str]:
    """Map (label, filename) -> original image path."""
    lookup = {}
    for path, label in samples:
        for name in (Path(path).name, Path(path).name + ".png"):
            key = (int(label), name)
            if key in lookup and lookup[key] != str(path):
                raise ValueError(f"Ambiguous original image filename: {name}")
            lookup[key] = str(path)
    return lookup


class DualInputBrainTumorDataset(Dataset):
    """Return a 6-channel tensor: [patch RGB (3), original RGB (3)]."""

    def __init__(
        self,
        patch_samples: List[Tuple[str, int]],
        original_lookup: Dict[Tuple[int, str], str],
        patch_transform: Optional[transforms.Compose] = None,
        original_transform: Optional[transforms.Compose] = None,
        grayscale: bool = False,
    ):
        self.samples = [(str(path), int(label)) for path, label in patch_samples if Path(path).exists()]
        self.original_lookup = original_lookup
        self.patch_transform = patch_transform
        self.original_transform = original_transform if original_transform is not None else patch_transform
        self.to_tensor = transforms.ToTensor()
        self.grayscale = grayscale
        dropped = len(patch_samples) - len(self.samples)
        if dropped > 0:
            import warnings
            warnings.warn(f"DualInputBrainTumorDataset: dropped {dropped} missing patch files")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        patch_path, label = self.samples[idx]
        patch_img = Image.open(patch_path).convert("RGB")
        original_path = self.original_lookup.get((label, Path(patch_path).name), patch_path)
        if not Path(original_path).exists():
            original_path = patch_path
        original_img = Image.open(original_path).convert("RGB")
        if self.grayscale:
            patch_img = patch_img.convert("L").convert("RGB")
            original_img = original_img.convert("L").convert("RGB")
        patch_tensor = self.patch_transform(patch_img) if self.patch_transform else self.to_tensor(patch_img)
        original_tensor = (
            self.original_transform(original_img)
            if self.original_transform
            else self.to_tensor(original_img)
        )
        return torch.cat([patch_tensor, original_tensor], dim=0), label
