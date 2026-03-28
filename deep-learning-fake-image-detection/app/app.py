from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

app = Flask(__name__)

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model.
# 1) Prefer new model saved by train.py at app/models/best_model.keras
# 2) Fallback to legacy models in ../models/ if present.
MODEL_PRIMARY = os.path.join(APP_DIR, "models", "best_model.keras")
MODEL_PATH_KERAS = os.path.join(APP_DIR, "..", "models", "fake_image_detector.keras")
MODEL_PATH_H5 = os.path.join(APP_DIR, "..", "models", "fake_image_detector_transfer.h5")

model = None
try:
    if os.path.exists(MODEL_PRIMARY):
        model = tf.keras.models.load_model(MODEL_PRIMARY)
    elif os.path.exists(MODEL_PATH_KERAS):
        model = tf.keras.models.load_model(MODEL_PATH_KERAS)
    elif os.path.exists(MODEL_PATH_H5):
        model = tf.keras.models.load_model(MODEL_PATH_H5)
    else:
        print(
            "Warning: No model file found. Place 'best_model.keras' in app/models "
            "or keep a compatible model in '../models'. Predictions will be disabled."
        )
except Exception as e:
    print(f"Warning: Failed to load model: {e}")
    model = None

UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Derive the expected input size from the loaded model to avoid
# mismatches between training and inference (e.g. 224 vs 300).
if model is not None and hasattr(model, "input_shape") and len(model.input_shape) == 4:
    TARGET_SIZE = (model.input_shape[1], model.input_shape[2])
else:
    TARGET_SIZE = (224, 224)
print("Using input size for prediction:", TARGET_SIZE)

# Tuned decision threshold for classifying an image as REAL.
# Obtained from tune_threshold.py using validation F1-score.
THRESHOLD = 0.44


def predict_image(img_path):
    """Return predicted label (Real/Fake) and confidence percentage."""
    if model is None:
        raise RuntimeError(
            "Model is not loaded. Place the model file in the 'models' folder."
        )
    img = image.load_img(img_path, target_size=TARGET_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prob = float(model.predict(img_array)[0][0])
    print("Raw probability:", prob)

    if prob > THRESHOLD:
        label = "Real"
        confidence = prob * 100.0
    else:
        label = "Fake"
        confidence = (1.0 - prob) * 100.0

    return label, round(confidence, 2)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    uploaded_image = None
    error = None
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            if model is None:
                error = (
                    "Model is not loaded. Upload is saved, "
                    "but prediction cannot be made."
                )
                uploaded_image = file.filename
            else:
                prediction, confidence = predict_image(filepath)
                uploaded_image = file.filename
            return render_template(
                "index.html",
                prediction=prediction,
                confidence=confidence,
                image=uploaded_image,
                error=error,
            )

    return render_template(
        "index.html", prediction=prediction, confidence=confidence, error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
