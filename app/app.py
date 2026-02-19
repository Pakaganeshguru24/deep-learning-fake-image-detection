from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load trained model (prefer .keras, fall back to transfer .h5)
MODEL_PATH_KERAS = os.path.join("..", "models", "fake_image_detector.keras")
MODEL_PATH_H5 = os.path.join("..", "models", "fake_image_detector_transfer.h5")

try:
    model = tf.keras.models.load_model(MODEL_PATH_KERAS)
except (OSError, ValueError):
    model = tf.keras.models.load_model(MODEL_PATH_H5)

UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def predict_image(img_path):
    """Return predicted label (Real/Fake) and confidence percentage."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prob = float(model.predict(img_array)[0][0])

    if prob > 0.5:
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
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            prediction, confidence = predict_image(filepath)
            uploaded_image = file.filename
            return render_template(
                "index.html",
                prediction=prediction,
                confidence=confidence,
                image=uploaded_image,
            )

    return render_template("index.html", prediction=prediction, confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True)
