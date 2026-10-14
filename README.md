# Fake Image Detection

A Flask web application that classifies uploaded images as **Real** or
**Fake** using a TensorFlow/Keras EfficientNetB3 model.

## Features

- Upload images for real/fake prediction
- Display the predicted label and confidence
- Drag-and-drop upload support
- Light and dark themes
- TensorFlow/Keras EfficientNetB3 inference
- Training and evaluation scripts
- Deterministic dataset splitting

## Requirements

- Python 3.11 is recommended.
- A TensorFlow-compatible system.
- Approximately 49.7 MB for the included trained model, plus space for Python
  dependencies.
- Windows PowerShell for the setup commands below.

## Setup and run

Clone the repository and enter the nested project directory:

```powershell
git clone https://github.com/Pakaganeshguru24/deep-learning-fake-image-detection.git
cd deep-learning-fake-image-detection
cd deep-learning-fake-image-detection
```

Create a virtual environment, install dependencies, and start the Flask app:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app\app.py
```

If PowerShell blocks virtual-environment activation, allow it for the current
terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Open the local application at:

<http://127.0.0.1:5000>

`127.0.0.1` is local-only. It is not a public hosted demo URL.

The trained model is included in the repository at:

```text
app/models/best_model.keras
```

The application accepts these image extensions:

```text
BMP, GIF, JPEG, JPG, PNG, WEBP, TIF, TIFF
```

Uploaded files are limited to 8 MB and are stored temporarily in:

```text
app/static/uploads/
```

## Dataset

The dataset is not included in the repository. Place a local dataset in this
structure before training:

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

Run the following commands from the nested
`deep-learning-fake-image-detection` directory.

### Create a deterministic split

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

### Train the model

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

### Tune the classification threshold

```powershell
python app\tune_threshold.py `
  --dataset-root dataset\strict_split `
  --model-path app\models\best_model.keras
```

The tuned threshold is saved at:

```text
app/models/threshold.json
```

### Evaluate the test split

```powershell
python app\eval_valid.py `
  --dataset-root dataset\strict_split `
  --split test `
  --model-path app\models\best_model.keras `
  --progress-every 100
```

### Run the complete pipeline

```powershell
python app\run_strict_pipeline.py --overwrite-split
```

## Project structure

```text
deep-learning-fake-image-detection/
├── app/
│   ├── app.py
│   ├── models/
│   │   └── best_model.keras
│   ├── templates/
│   │   └── index.html
│   ├── train.py
│   ├── build_stratified_splits.py
│   ├── tune_threshold.py
│   ├── eval_valid.py
│   └── run_strict_pipeline.py
├── requirements.txt
├── .python-version
└── README.md
```

Configuration used by the Flask application is currently defined in
`app.py`; there is no separate `config.py` file in this revision.

## Important notes

- Predictions are model-based estimates.
- This project is not a definitive forensic or legal tool.
- The dataset is not included.
- Runtime uploads, generated reports, virtual environments, and dataset files
  are excluded from version control.
- Flask's development server should not be used for production deployment.

## License

This project is licensed under the MIT License. See the `LICENSE` file for
details.
