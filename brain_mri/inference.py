"""Single-image segmentation inference shared by training and the app."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def predict_mask(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
    ensure_nonempty: bool = False,
) -> np.ndarray:
    """Run U-Net inference on a single grayscale image (H, W) → binary mask (H, W).

    The image is resized to 256×256 for the network, then the mask is
    resized back to the original spatial dimensions.

    Empty predictions remain empty by default. Set ``ensure_nonempty=True``
    only when the caller already knows a lesion is present; forcing a mask on
    an unknown image creates false-positive ROIs for no-tumor inputs.
    """
    model.eval()
    h, w = image.shape[:2]

    img = image.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    img_tensor = F.interpolate(img_tensor, size=(256, 256), mode="bilinear", align_corners=False)
    img_tensor = img_tensor.to(device)

    logits = model(img_tensor)
    prob_256 = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)

    from PIL import Image as PILImage

    # Resize probability map back first (bilinear), then threshold at full resolution.
    prob_full = np.array(
        PILImage.fromarray(prob_256, mode="F").resize((w, h), resample=PILImage.BILINEAR),
        dtype=np.float32,
    )
    mask_full = (prob_full > threshold).astype(np.uint8)

    # Optional fallback for known-positive images only.
    if ensure_nonempty and mask_full.sum() == 0 and float(prob_full.max()) > 0.0:
        adaptive_thr = float(np.clip(prob_full.max() * 0.5, 0.10, max(0.10, threshold - 1e-3)))
        mask_full = (prob_full > adaptive_thr).astype(np.uint8)

    if ensure_nonempty and mask_full.sum() == 0:
        q = float(np.quantile(prob_full, 0.995))
        mask_full = (prob_full >= q).astype(np.uint8)

    # Keep only largest connected component to reduce speckle noise.
    if mask_full.sum() > 0:
        try:
            from scipy import ndimage
            labeled, ncomp = ndimage.label(mask_full)
            if ncomp > 1:
                sizes = ndimage.sum(mask_full, labeled, index=np.arange(1, ncomp + 1))
                largest = int(np.argmax(sizes) + 1)
                mask_full = (labeled == largest).astype(np.uint8)
        except Exception:
            pass

    return mask_full
