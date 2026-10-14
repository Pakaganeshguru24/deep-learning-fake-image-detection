# Fake Image Detection

A Flask web application that classifies an uploaded image as **Real** or
**Fake** using a trained TensorFlow/Keras EfficientNet model.

## Run the web app

The application needs a trained model at:

```text
app/models/best_model.keras
```

The model is intentionally excluded from Git because model files and datasets
are large. Keep a copy in that location locally, or provide the model through
your team's artifact storage before starting the app.

### Windows PowerShell

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app\app.py
```

Open <http://127.0.0.1:5000> in a browser. Upload a JPG, JPEG, PNG, or other
supported image format and select **Analyze image**.

If PowerShell blocks virtual-environment activation, run this once in the
current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

The repository may contain a pre-existing local `.venv310` environment. It is
not required for new setups; use the commands above to create a fresh one.

## Train and evaluate

Training requires a local dataset with this structure:

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

These data files are intentionally ignored by Git. To create a deterministic
strict split from existing splits, run:

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

Then train:

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

Tune the validation threshold and evaluate the held-out test split:

```powershell
python app\tune_threshold.py `
  --dataset-root dataset\strict_split `
  --model-path app\models\best_model.keras

python app\eval_valid.py `
  --dataset-root dataset\strict_split `
  --split test `
  --model-path app\models\best_model.keras `
  --progress-every 100
```

The complete split, train, tune, and evaluation sequence can also be run with:

```powershell
python app\run_strict_pipeline.py --overwrite-split
```

## Project structure

```text
app/
├── app.py                    # Flask web application and inference
├── templates/index.html      # Upload and prediction UI
├── train.py                 # Model training
├── build_stratified_splits.py
├── tune_threshold.py
├── eval_valid.py
└── run_strict_pipeline.py   # End-to-end training pipeline
requirements.txt              # Reproducible runtime/training dependencies
```

Runtime uploads are written to `app/static/uploads/`. Logs, datasets, model
checkpoints, virtual environments, and generated reports are local artifacts
and are excluded from version control.
