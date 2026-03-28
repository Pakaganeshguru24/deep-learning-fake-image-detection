import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

# Paths relative to this app folder
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
MODEL_PATH = os.path.join(APP_DIR, "models", "best_model.keras")
VAL_PATH = os.path.join(PROJECT_ROOT, "dataset", "valid")

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

BATCH_SIZE = 32

# Validation data
print("Using IMG_SIZE =", IMG_SIZE)
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_data = datagen.flow_from_directory(
    VAL_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False,
)

# Get predictions
probs = model.predict(val_data).ravel()
y_true = val_data.classes

# Try thresholds from 0.1 to 0.9
thresholds = np.arange(0.1, 0.9, 0.01)

best_threshold = 0.5
best_f1 = 0.0

for t in thresholds:
    y_pred = (probs > t).astype(int)
    f1 = f1_score(y_true, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print("Best Threshold:", round(best_threshold, 3))
print("Best F1 Score:", round(best_f1, 4))
