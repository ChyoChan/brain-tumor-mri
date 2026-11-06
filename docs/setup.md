# Reproducible setup and weights

Use Python 3.11. Install into a fresh environment rather than upgrading an
unrelated Conda environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-runtime.txt
python scripts/run_frontend.py
```

`requirements-runtime.txt` pins direct frontend dependencies. The core pins come
from the local inference environment; `grad-cam` is the correct distribution
name for `pytorch_grad_cam`. Grad-CAM declares `opencv-python`, while
Albumentations declares `opencv-python-headless`; training installs both at the
same pinned OpenCV version. The project uses image-processing APIs, not OpenCV
GUI windows. Minimal Linux systems may need their distribution's `libGL` runtime
package when installing the frontend environment alone.

For training, evaluation plots, and notebook utilities:

```bash
python -m pip install -r requirements-training.txt
```

For tests, add `requirements-dev.txt`; `requirements.txt` installs all three
groups. These are direct dependency pins, not a complete transitive lock file.
The pinned requirements resolve in a clean-install dry run on the development
Mac without modifying its existing environment. GitHub Actions installs the
training extras on Linux and checks dependency compatibility, tests, and Python
compilation without downloading datasets or project checkpoints.
CUDA users should install matching PyTorch wheels from the official PyTorch
instructions instead of using the CPU-only CI installation command.

## Existing checkpoints

Model weights are not included in Git. `models/checkpoints.json` records the
size and SHA-256 of the five original checkpoints used by the frontend. It is a
download-integrity manifest, separate from each model's inference metadata.
To verify weights already on disk:

```bash
python scripts/download_checkpoints.py --verify-only
```

The manifest deliberately has `url: null` entries because there are no published
download URLs supplied with this repository. A fresh clone therefore cannot run
live inference until the owner provides the weights or the user trains models.
An absent or mismatched checkpoint is reported, and the app identifies missing
models in its health status.

To publish an approved checkpoint as the repository owner:

1. Confirm the file may be redistributed under its data/model license.
2. Upload that `.pth` file as a GitHub Release asset.
3. Copy the asset's HTTPS download URL into the matching manifest entry.
4. Confirm its size and checksum still match the manifest.

Users can then run:

```bash
python scripts/download_checkpoints.py
```

Downloads are streamed to a temporary file and checked against both size and
SHA-256 before an existing checkpoint is replaced. Weights without a published
URL produce a clear error. Do not put access tokens in manifest URLs.

Newly trained weights need their own published asset, hash entry, and model
metadata; they must not be substituted under an old checkpoint's checksum.

## Advanced training and evaluation

### Frozen classification splits

Create a split once before classifier training and reuse it across model modes:

```bash
python scripts/prepare_classification_split.py --data-dir data/raw --output data/processed/classification_split_v1.json
```

Existing manifests are never overwritten. The loader checks dataset changes and
keeps byte-identical images together. Add `--groups-csv patients.csv` when creating
the manifest to also group by patient. Supply `path,patient_id` columns, paths
relative to `--data-dir`, and a row for every image. Without those identifiers,
patient isolation and detection of re-encoded or augmented copies are not
established. The source dataset's `Training`/`Testing` folders are combined before
this new split is created; their names do not define the frozen evaluation split.

### Cropped and dual-input classifiers

Install `requirements-training.txt` before using the training/evaluation scripts.
For `--mode cropped` or `--mode dual`, `segmentation.model`, `in_channels`,
`out_channels`, and `base_filters` in `configs/pipeline_config.yaml` must match
the segmentation weights. `--seg-checkpoint` changes the weight file, not its
architecture. With the default U-Net configuration:

```bash
python scripts/train_segmentation.py --model unet --save-path models/segmentation/best_unet_real_mask.pth
python scripts/train_classification.py --mode dual --manifest data/processed/classification_split_v1.json --seg-checkpoint models/segmentation/best_unet_real_mask.pth
python scripts/eval_classification.py --mode dual --manifest data/processed/classification_split_v1.json --seg-model unet --seg-checkpoint models/segmentation/best_unet_real_mask.pth --output outputs/metrics/dual_evaluation.json
```

The explicit U-Net save path is recognized by the app; the training script's
default `best_unet.pth` is not. ResUNet and Attention U-Net training defaults
already match the app's checkpoint names. For those architectures, update the
classifier training configuration and evaluation's `--seg-model` accordingly.

New classifiers save `best_resnet50_<mode>.pth` plus a `.metadata.json` sidecar.
Keep them together: metadata records class order, grayscale/ImageNet
preprocessing, split fingerprints, and weight checksums. Cropped/dual metadata
also records the segmentation model and checksum; inference rejects a different
segmenter. The app prefers new full/dual checkpoints over legacy weights.
Cropped-only inference is available through the evaluation CLI, not the UI.

New `roi_v1` classifiers use label-independent crops. The original dual-input
checkpoint used full images in both views for no-tumor training examples.
Its serving path uses a separate full-image classifier as a no-tumor gate to
reproduce that legacy contract without consulting the true label. New `roi_v1`
weights do not use this gate.

### Confidence calibration

Fit temperature scaling on validation data, then evaluate once on test data.
Fitting requires a newly trained checkpoint with metadata tied to the same
manifest; the script rejects test-set fitting and unverified legacy weights.
The shared evaluator still runs segmentation in full-image mode. The examples
below therefore also require the default ResUNet checkpoint; select another
installed segmenter with `--seg-model`/`--seg-checkpoint` if needed:

```bash
python scripts/eval_classification.py --manifest data/processed/classification_split_v1.json --split val --mode full --fit-temperature models/classification/full_temperature.json --output outputs/metrics/full_validation.json
python scripts/eval_classification.py --manifest data/processed/classification_split_v1.json --split test --mode full --temperature models/classification/full_temperature.json --output outputs/metrics/full_calibrated_test.json
MRI_CALIBRATION=models/classification/full_temperature.json python scripts/run_frontend.py
```

Calibration artifacts must match the classifier, split, class order, and any
segmenter/gate weights. Without an artifact, displayed percentages are
uncalibrated softmax scores. Temperature scaling does not change the predicted
class or establish clinical reliability. Restart the server after replacing
weights, metadata, or calibration files; runtime settings and models are cached.

### Comparing new and archived experiments

Current segmentation training uses grayscale inputs in `[0, 1]`, matching serving.
Earlier shared training transforms used `[-1, 1]`; archived metrics describe
those historical runs and must not be presented as scores for retrained weights.
The main notebook now generates lossless, label-independent crops under
`data/processed/patches_resunet_roi_v1/`; reruns do not reuse its older crops.

Segmentation training and evaluation automatically fall back to Otsu pseudo-masks
when real image/mask pairs are absent. Provide expert masks for evaluation
against annotations; pseudo-mask scores do not measure that agreement.
Classification evaluation writes accuracy, macro F1, per-class recall, confusion
matrices, timing, ECE/NLL, predictions, and reliability CSV/PNG files beside the
output JSON. Its timing covers prediction without UI reports, Grad-CAM, or
preview encoding. Retraining and new splits can change all archived scores.

## Automated checks

```bash
python -m pip install -r requirements.txt
python -m pip check
PYTHONPATH=. python -m pytest tests -q
python -m compileall -q brain_mri app scripts
```

`.github/workflows/tests.yml` runs these checks on pushes and pull requests using
CPU PyTorch. Tests use small generated images and model doubles, so the private
dataset and full checkpoint files are unnecessary.
