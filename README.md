# Fake Image Detection

This repository contains a nested project directory:

```text
deep-learning-fake-image-detection/
```

All Python commands below must be run from that nested directory.

## Clone and enter the project (Windows PowerShell)

```powershell
git clone https://github.com/Pakaganeshguru24/deep-learning-fake-image-detection.git
cd deep-learning-fake-image-detection\deep-learning-fake-image-detection
```

## Required model file (not in Git)

Inference requires this file:

```text
app/models/best_model.keras
```

The model is intentionally excluded from Git, so you must provide it separately.

## Local development vs public demo

- `http://127.0.0.1:5000` is a **local development URL** on your own machine.
- It is **not** a hosted public demo.
- To provide a public demo, deploy the Flask app and share the deployed URL.

## Run the Flask app locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app\app.py
```

Open <http://127.0.0.1:5000>.

Supported upload extensions: **BMP, GIF, JPG, JPEG, PNG, WEBP, TIF, TIFF**.

Optional local debug mode:

```powershell
$env:FLASK_DEBUG = "1"
python app\app.py
```

## Training, threshold tuning, and evaluation

Dataset files are intentionally excluded from Git. Place your local dataset under:

```text
dataset/
├── train/
│   ├── fake/
│   └── real/
├── valid/
│   ├── fake/
│   └── real/
└── test/
    ├── fake/
    └── real/
```

### 1) Build strict split

```powershell
python app\build_stratified_splits.py `
  --dataset-root dataset `
  --source-mode existing-splits `
  --output-dir dataset\strict_split `
  --seed 42 `
  --valid-ratio 0.15 `
  --test-ratio 0.10 `
  --link-mode hardlink `
  --overwrite
```

### 2) Train model

```powershell
python app\train.py `
  --dataset-root dataset\strict_split `
  --model-path app\models\best_model.keras `
  --seed 42 `
  --img-size 300 `
  --batch-size 16 `
  --epochs-phase1 10 `
  --epochs-phase2 6
```

### 3) Tune threshold and persist artifact

```powershell
python app\tune_threshold.py `
  --dataset-root dataset\strict_split `
  --model-path app\models\best_model.keras `
  --output-json app\models\threshold.json
```

If `app/models/threshold.json` exists, inference/evaluation use it; otherwise they fall back to shared defaults.

### 4) Evaluate

```powershell
python app\eval_valid.py `
  --dataset-root dataset\strict_split `
  --split test `
  --model-path app\models\best_model.keras `
  --threshold-config app\models\threshold.json `
  --progress-every 100
```

### 5) End-to-end strict pipeline

```powershell
python app\run_strict_pipeline.py --overwrite-split
```

The strict pipeline runs split → train → tune (writes threshold JSON) → test evaluation.
