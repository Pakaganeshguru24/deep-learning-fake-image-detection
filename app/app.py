from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from keras.utils import load_img, img_to_array
from keras.applications.efficientnet import preprocess_input
from werkzeug.utils import secure_filename
import os
import time
import json
import tempfile
import zipfile

app = Flask(__name__)

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model.
# 1) Prefer new model saved by train.py at app/models/best_model.keras
# 2) Fallback to legacy models in ../models/ if present.
MODEL_PRIMARY = os.path.join(APP_DIR, "models", "best_model.keras")
MODEL_PATH_KERAS = os.path.join(APP_DIR, "..", "models", "fake_image_detector.keras")
MODEL_PATH_H5 = os.path.join(APP_DIR, "..", "models", "fake_image_detector_transfer.h5")

model = None


def load_model_compatibly(model_path):
    """Load models saved by older Keras versions when possible."""
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as first_error:
        temporary_path = None
        try:
            with zipfile.ZipFile(model_path) as source:
                fd, temporary_path = tempfile.mkstemp(suffix=".keras")
                os.close(fd)
                with zipfile.ZipFile(temporary_path, "w") as target:
                    for item in source.infolist():
                        data = source.read(item.filename)
                        if item.filename == "config.json":
                            config = json.loads(data)

                            def remove_unsupported_config(value):
                                if isinstance(value, dict):
                                    if value.get("quantization_config") is None:
                                        value.pop("quantization_config", None)
                                    for child in value.values():
                                        remove_unsupported_config(child)
                                elif isinstance(value, list):
                                    for child in value:
                                        remove_unsupported_config(child)

                            remove_unsupported_config(config)
                            data = json.dumps(config).encode("utf-8")
                        target.writestr(item, data)
            return tf.keras.models.load_model(temporary_path, compile=False)
        except Exception:
            raise first_error
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)


try:
    model_path = next(
        (
            path
            for path in (MODEL_PRIMARY, MODEL_PATH_KERAS, MODEL_PATH_H5)
            if os.path.exists(path)
        ),
        None,
    )
    if model_path:
        model = load_model_compatibly(model_path)
    else:
        print(
            "Warning: No model file found. Place 'best_model.keras' in app/models "
            "or keep a compatible model in '../models'. Predictions will be disabled."
        )
except Exception as e:
    print(f"Warning: Failed to load model: {e}")
    model = None

UPLOAD_FOLDER = os.path.join(APP_DIR, "static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Derive the expected input size from the loaded model to avoid
# mismatches between training and inference (e.g. 224 vs 300).
if model is not None and hasattr(model, "input_shape") and len(model.input_shape) == 4:
    TARGET_SIZE = (model.input_shape[1], model.input_shape[2])
else:
    TARGET_SIZE = (224, 224)
print("Using input size for prediction:", TARGET_SIZE)

# Tuned decision threshold for classifying an image as REAL.
# Obtained from tune_threshold.py using validation F1-score.
THRESHOLD = 0.47
REAL_IF_HIGH_PROB = True
UNCERTAINTY_MARGIN = 0.08
TTA_STD_MAX = 0.12
ENABLE_RUNTIME_CALIBRATION = False


def calibrate_with_validation_data():
    """Calibrate threshold and label direction from validation data if available."""
    global THRESHOLD, REAL_IF_HIGH_PROB
    if model is None:
        return

    valid_dir = os.path.join(APP_DIR, "..", "dataset", "strict_split", "valid")
    if not os.path.isdir(valid_dir):
        valid_dir = os.path.join(APP_DIR, "..", "dataset", "valid")
    if not os.path.isdir(valid_dir):
        print("Validation folder not found; using default threshold:", THRESHOLD)
        return

    try:
        raw_val_data = tf.keras.utils.image_dataset_from_directory(
            valid_dir,
            labels="inferred",
            label_mode="int",
            image_size=TARGET_SIZE,
            batch_size=32,
            shuffle=False,
        )

        val_data = raw_val_data.map(
            lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), tf.cast(y, tf.float32)),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)

        probs = model.predict(val_data, verbose=0).ravel()
        y_true = np.concatenate([y.numpy() for _, y in raw_val_data], axis=0).astype(int)

        thresholds = np.arange(0.1, 0.91, 0.01)
        best_acc = -1.0
        best_t = THRESHOLD
        best_real_if_high = True

        for t in thresholds:
            pred_normal = (probs > t).astype(int)
            acc_normal = float((pred_normal == y_true).mean())

            pred_inverted = 1 - pred_normal
            acc_inverted = float((pred_inverted == y_true).mean())

            if acc_normal > best_acc:
                best_acc = acc_normal
                best_t = float(t)
                best_real_if_high = True

            if acc_inverted > best_acc:
                best_acc = acc_inverted
                best_t = float(t)
                best_real_if_high = False

        THRESHOLD = best_t
        REAL_IF_HIGH_PROB = best_real_if_high
        print(
            "Calibration complete. threshold=",
            round(THRESHOLD, 3),
            "real_if_high_prob=",
            REAL_IF_HIGH_PROB,
            "val_acc=",
            round(best_acc * 100.0, 2),
            "%",
        )
    except Exception as e:
        print(f"Calibration skipped due to error: {e}")


if ENABLE_RUNTIME_CALIBRATION:
    calibrate_with_validation_data()
else:
    print("Runtime threshold calibration disabled. Using fixed threshold:", THRESHOLD)


def predict_image(img_path):
    """Return predicted label, confidence, and diagnostics for UI display."""
    if model is None:
        raise RuntimeError(
            "Model is not loaded. Place the model file in the 'models' folder."
        )

    img = load_img(img_path, target_size=TARGET_SIZE)
    base = img_to_array(img)

    # Keep inference consistent with eval_valid.py and tune_threshold.py: single-view prediction.
    batch = np.expand_dims(base, axis=0).astype(np.float32)
    batch = preprocess_input(batch)

    decision_prob = float(model.predict(batch, verbose=0).ravel()[0])
    prob_mean = decision_prob
    prob_median = decision_prob
    prob_std = 0.0
    print("Prediction prob:", decision_prob)
    predicts_real = decision_prob > THRESHOLD if REAL_IF_HIGH_PROB else decision_prob <= THRESHOLD

    prob_real = decision_prob if REAL_IF_HIGH_PROB else (1.0 - decision_prob)
    prob_fake = 1.0 - prob_real

    if predicts_real:
        label = "Real"
        confidence = prob_real * 100.0
    else:
        label = "Fake"
        confidence = prob_fake * 100.0

    note = f"Model prob: {decision_prob:.3f}"
    return label, round(confidence, 2), note


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    prediction_note = None
    uploaded_image = None
    error = None
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            safe_name = secure_filename(file.filename)
            stamped_name = f"{int(time.time())}_{safe_name}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], stamped_name)
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
            file.save(filepath)
            static_relpath = os.path.join("uploads", stamped_name).replace("\\", "/")
            if model is None:
                error = (
                    "Model is not loaded. Upload is saved, "
                    "but prediction cannot be made."
                )
                uploaded_image = static_relpath
            else:
                try:
                    prediction, confidence, prediction_note = predict_image(filepath)
                except Exception as exc:
                    error = f"Prediction failed: {exc}"
                uploaded_image = static_relpath
            return render_template(
                "index.html",
                prediction=prediction,
                confidence=confidence,
                prediction_note=prediction_note,
                image=uploaded_image,
                error=error,
            )

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        prediction_note=prediction_note,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
