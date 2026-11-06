# Brain Tumor MRI Analysis Pipeline

[![Python checks](https://github.com/ChyoChan/brain-tumor-mri/actions/workflows/tests.yml/badge.svg)](https://github.com/ChyoChan/brain-tumor-mri/actions/workflows/tests.yml)

A PyTorch research project for exploring brain MRI segmentation, four-class
classification, and model explanations through a local web interface. Developed
as a COMP4471 term project.

**Checkpoint files and MRI images are not included.** A fresh clone can open the
interface, but inference requires trained models. Checkpoint download URLs have
not been published; obtain the original weights from the repository owner or
[train your own](#train-and-evaluate).

## What the project does

Each analysis runs the selected models on the supplied image. The saved CSV/JSON
metrics are experiment records, not the source of frontend predictions.

| Stage | Choices | Output |
| --- | --- | --- |
| Segmentation | U-Net, ResUNet, Attention U-Net | Predicted mask and region-of-interest (ROI) crop |
| Classification | Full-image ResNet50 or dual-input ResNet50 (crop + original) | Scores for glioma, meningioma, pituitary, and no tumor |
| Explanation | Grad-CAM and an optional predefined text summary | Attention overlay and measured model outputs |

The UI defaults to **ResUNet + Full image**. Segmentation still runs in this mode
to produce the mask and crop displayed by the viewer. Dual-input mode uses a
full-image reference Grad-CAM; it does not directly explain the fused dual model.
The summary runs locally without a hosted language model.

This is research and educational software. Scores and attention maps do not
establish a clinical diagnosis; scores are uncalibrated unless a compatible
calibration artifact is configured.

## Run locally

Use **Python 3.11** and Git. The commands below use a macOS/Linux shell. On
Windows, create the environment with `py -3.11 -m venv .venv` and activate it with
`.\.venv\Scripts\Activate.ps1`; the subsequent Python commands are the same.

### 1. Clone and install

```bash
git clone https://github.com/ChyoChan/brain-tumor-mri.git
cd brain-tumor-mri
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-runtime.txt
```

The runtime selects CUDA when available, then Apple MPS, then CPU. A GPU is
optional. See [setup details](docs/setup.md) for dependency and CUDA notes.

### 2. Add trained models

For the default **ResUNet + Full image** settings, place these original weights
under the project root:

```text
models/
├── segmentation/
│   └── best_resunet.pth
└── classification/
    └── pure_baseline/
        └── best_resnet50_pure_classification.pth
```

A newly trained full-image classifier can instead use
`models/classification/best_resnet50_full.pth` with its matching
`best_resnet50_full.metadata.json`. Keep the metadata beside the weights. New
classifier checkpoints take priority over the original fallback files.

Other model selections need their corresponding checkpoints. The complete list
of five original files, sizes, and SHA-256 checksums is in
[models/checkpoints.json](models/checkpoints.json). If you have **all five original
files**, verify them with:

```bash
python scripts/download_checkpoints.py --verify-only
```

This command checks the full original set, not just the two default models or
newly trained weights. No automatic download is available while manifest URLs
are `null`. See [checkpoint setup](docs/setup.md#existing-checkpoints).

### 3. Start the frontend

```bash
python scripts/run_frontend.py
```

Open **[http://127.0.0.1:8765](http://127.0.0.1:8765)**. The server binds to your
local machine. Keep the terminal open; press **Ctrl+C** there to stop it.

If the port is occupied, select another port:

```bash
python scripts/run_frontend.py --port 8766
```

Then open `http://127.0.0.1:8766`. On macOS/Linux with `lsof` and `ps` available,
you can also stop this project's server from another activated terminal:

```bash
python scripts/stop_frontend.py --port 8766
```

The default stop port is 8765. The helper refuses to terminate unrelated services.

### 4. Analyze an image

1. Upload a PNG, JPG, or TIFF image, or choose an available local sample.
2. Select the segmentation model and classifier whose weights you installed.
3. Leave **Generate report** checked for the text summary, or uncheck it to skip
   that step. Segmentation, classification, and Grad-CAM still run.
4. Run the pipeline and inspect the original image, mask, ROI crop, class scores,
   attention overlay, and analysis details.

Uploads support individual image slices up to **20 MB and 25 million pixels**.
The UI does not read DICOM or NIfTI volumes. No dataset download is needed when
you upload your own supported image and have the required weights.

To enable **Browse local images**, use this optional layout:

```text
data/raw/Classification/Testing/
├── glioma/
├── meningioma/
├── pituitary/
└── notumor/
```

Place images inside the class folders, then restart the server. The browser
supports class filters, filename search, and pagination. Folder labels are
dataset annotations, not predictions. The four quick-sample cards require the
original sample filenames; the browser works with other supported filenames.

### Optional: macOS Conda launchers

The Finder launchers require an **existing Conda environment** with the runtime
requirements installed. They default to an environment named `rag`; set
`MRI_CONDA_ENV` to use your own environment:

```bash
MRI_CONDA_ENV=myenv ./scripts/start_frontend.command
MRI_CONDA_ENV=myenv ./scripts/stop_frontend.command
```

Both accept `--port`. They locate the project automatically and do not create an
environment or install dependencies. The Python commands above work with the
virtual environment used in the quick start.

## Train and evaluate

The main notebook downloads the combined
[brain tumor segmentation and classification dataset](https://www.kaggle.com/datasets/indk214/brain-tumor-dataset-segmentation-and-classification).
Obtain and extract the data separately under `data/raw/`, retaining its
`Classification/` and `Segmentation/` folders. Dataset access and reuse follow
the source's terms. Training settings live in
[configs/pipeline_config.yaml](configs/pipeline_config.yaml).

Install the training extras, freeze a classification split, and train models
that the default frontend can discover:

```bash
python -m pip install -r requirements-training.txt
python scripts/prepare_classification_split.py --data-dir data/raw --output data/processed/classification_split_v1.json
python scripts/train_segmentation.py --model resunet
python scripts/train_classification.py --mode full --manifest data/processed/classification_split_v1.json
python scripts/eval_classification.py --manifest data/processed/classification_split_v1.json --mode full --output outputs/metrics/full_evaluation.json
```

This saves `models/segmentation/best_resunet.pth` and
`models/classification/best_resnet50_full.pth` with its metadata sidecar. The
classifier's initial pretrained ImageNet weights may download on the first
training run. Full-image classifier training does not need a segmenter. The
supplied evaluator and frontend also run segmentation, so they need the selected
segmenter's weights.

For segmentation, provide real image/mask pairs: class folders with
`scan.png` / `scan_mask.png`, or parallel `images/` and `masks/` trees. The training
script falls back to Otsu pseudo-masks if real pairs are absent; those are not
expert annotations.

The frozen split keeps byte-identical images together. Patient isolation needs
a complete `path,patient_id` CSV supplied with `--groups-csv`; exact-file hashing
does not detect re-encoded or augmented duplicates. Existing split manifests are
not overwritten. A file's location under the UI's local `Testing/` folder does
not establish that it belongs to the frozen test split.

Evaluation exports accuracy, macro F1, per-class recall, confusion matrices,
calibration measures, timing, and per-image results. See
[advanced training and calibration notes](docs/setup.md#advanced-training-and-evaluation)
for cropped/dual models, checkpoint compatibility, and validation-only
temperature scaling. [Experiment notebooks](notebooks/README.md) use the same
shared Python package.

## Results and limitations

The [archived full-image baseline](outputs/metrics/resnet50_full_test_metrics.json)
records **99.15% accuracy**. Saved results describe the original experiments;
retraining with corrected preprocessing and a newly frozen split may produce
different scores. Patient-independent validation has not been established.

[The dataset audit](outputs/metrics/dataset_audit.json) found 426 extra
byte-identical files among 7,023 images. Current split generation groups exact
duplicates, but this does not retroactively validate the historical scores.
Segmentation results also depend on whether real or pseudo-masks were used.
See [saved metrics](outputs/metrics) and the
[results summary](outputs/results_summary.csv) for the archived outputs.

## Project structure

```text
app/                     FastAPI server and browser UI
brain_mri/               shared inference, data, models, training, and explanations
configs/                 pipeline configuration
scripts/                 start/stop, training, evaluation, and checkpoint commands
notebooks/               experiment notebooks importing brain_mri/
models/checkpoints.json  original checkpoint size/checksum manifest
outputs/metrics/         archived metrics; vlm/ contains historical VLM experiments
docs/setup.md            dependency, model compatibility, and calibration details
```

Raw data, generated patches, model weights, reports, and generated figures stay
local and are excluded by [.gitignore](.gitignore).

## Development checks

From the project root with the virtual environment activated:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -m pytest tests -q
python -m compileall -q brain_mri app scripts
```

`requirements-runtime.txt` installs frontend dependencies;
`requirements-training.txt` adds training/notebook tools;
`requirements-dev.txt` adds test tools. `requirements.txt` includes all three.
[GitHub Actions](https://github.com/ChyoChan/brain-tumor-mri/actions/workflows/tests.yml)
checks dependencies, tests, and Python compilation on Linux with CPU PyTorch.
Tests use generated fixtures and model doubles without private data or weights.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| UI opens, but analysis reports a missing checkpoint | Add both the selected segmenter and classifier weights at the paths above; the interface can open without them. |
| Address already in use | Stop the existing server or start with `--port 8766` and use that URL. |
| No samples appear | Upload an image, or add images under the optional `Classification/Testing/` layout and restart. No MRI samples ship with the repo. |
| Metadata or calibration mismatch | Keep sidecars with their matching weights; select the recorded training segmenter and restart after replacing files. |
| A training/test import is missing | Install `requirements-training.txt` or the complete `requirements.txt` in the same environment used to run the command. |
