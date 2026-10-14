import argparse
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, recall_score

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
preprocess_input = tf.keras.applications.efficientnet.preprocess_input

# Paths relative to this app folder
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
DEFAULT_MODEL_PATH = os.path.join(APP_DIR, "models", "best_model.keras")
DEFAULT_DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")


def parse_args():
    parser = argparse.ArgumentParser(description="Tune classification threshold on validation split.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


args = parse_args()
MODEL_PATH = os.path.abspath(args.model_path)
VAL_PATH = os.path.join(os.path.abspath(args.dataset_root), "valid")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first.")
if not os.path.isdir(VAL_PATH):
    raise FileNotFoundError(f"Validation folder not found at {VAL_PATH}.")

# Load model
print("Loading model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

# Derive image size from model to stay in sync with training
if hasattr(model, "input_shape") and len(model.input_shape) == 4:
    IMG_SIZE = model.input_shape[1]
else:
    IMG_SIZE = 300

BATCH_SIZE = int(args.batch_size)

# Validation data
print("Using IMG_SIZE =", IMG_SIZE)
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_data = datagen.flow_from_directory(
    VAL_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['fake', 'real'],
    shuffle=False,
)

expected_class_indices = {'fake': 0, 'real': 1}
if val_data.class_indices != expected_class_indices:
    raise RuntimeError(
        f"Unexpected class mapping: {val_data.class_indices}. Expected: {expected_class_indices}"
    )

# Get predictions
probs = model.predict(val_data).ravel()
y_true = val_data.classes


def _macro_f1(y_true_arr, y_pred_arr):
    f1_fake = float(f1_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0))
    f1_real = float(f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    return (f1_real + f1_fake) / 2.0, f1_real, f1_fake


# Try thresholds from 0.1 to 0.9
thresholds = np.arange(0.1, 0.9, 0.01)

best_threshold = 0.5
best_macro_f1 = 0.0
best_real_f1 = 0.0
best_fake_f1 = 0.0
best_real_recall = 0.0

for t in thresholds:
    y_pred = (probs > t).astype(int)
    macro_f1, real_f1, fake_f1 = _macro_f1(y_true, y_pred)
    real_recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        best_threshold = float(t)
        best_real_f1 = real_f1
        best_fake_f1 = fake_f1
        best_real_recall = real_recall

print("Best Threshold (macro F1):", round(best_threshold, 3))
print("Macro F1:", round(best_macro_f1, 4))
print("Real F1:", round(best_real_f1, 4), "| Fake F1:", round(best_fake_f1, 4))
print("Real Recall:", round(best_real_recall, 4))
