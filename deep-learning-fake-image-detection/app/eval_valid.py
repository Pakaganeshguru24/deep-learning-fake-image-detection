import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

# Match paths and settings from train.py
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")
VALID_DIR = os.path.join(DATASET_ROOT, "valid")

MODEL_PATH = os.path.join(APP_DIR, "models", "best_model.keras")

# Use the same tuned threshold as the web app (see app.py / tune_threshold.py)
THRESHOLD = 0.44

if not os.path.isdir(VALID_DIR):
    raise FileNotFoundError(
        f"Expected validation folder at '{VALID_DIR}' with 'real' and 'fake' subfolders."
    )

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Could not find trained model at '{MODEL_PATH}'. Run train.py first."
    )

print("Loading model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

# Derive expected input size from model
if hasattr(model, "input_shape") and len(model.input_shape) == 4:
    img_size = (model.input_shape[1], model.input_shape[2])
else:
    img_size = (300, 300)

print("Using validation image size:", img_size)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

val_data = val_datagen.flow_from_directory(
    VALID_DIR,
    target_size=img_size,
    batch_size=32,
    class_mode="binary",
    shuffle=False,
)

# Evaluate accuracy on the whole validation set
loss, acc = model.evaluate(val_data, verbose=1)
print("\nValidation loss:", loss)
print("Validation accuracy:", acc * 100.0, "%")

# Optional: show basic confusion info
probs = model.predict(val_data, verbose=1).ravel()
preds = (probs > THRESHOLD).astype(int)
true = val_data.classes

# Build 2x2 confusion matrix
# rows = true (fake=0, real=1), cols = pred (fake=0, real=1)
cm = np.zeros((2, 2), dtype=int)
for t, p in zip(true, preds):
    cm[int(t), int(p)] += 1

tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

print("\nConfusion Matrix (counts):")
print("            Pred Fake   Pred Real")
print(f"True Fake   {tn:9d} {fp:10d}")
print(f"True Real   {fn:9d} {tp:10d}")

acc_manual = (preds == true).mean() * 100.0
print("\nManual accuracy check:", acc_manual, "%")
