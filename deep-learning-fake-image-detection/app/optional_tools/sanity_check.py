import argparse
import os
import numpy as np
import tensorflow as tf

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
preprocess_input = tf.keras.applications.efficientnet.preprocess_input

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
MODEL_PATH = os.path.join(APP_DIR, "models", "best_model.keras")
VALID_DIR = os.path.join(PROJECT_ROOT, "dataset", "valid")


def _metrics(y_true, y_pred):
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    acc = float((y_pred == y_true).mean())

    precision_fake = tn / (tn + fn) if (tn + fn) else 0.0
    recall_fake = tn / (tn + fp) if (tn + fp) else 0.0
    f1_fake = (
        2 * precision_fake * recall_fake / (precision_fake + recall_fake)
        if (precision_fake + recall_fake)
        else 0.0
    )

    precision_real = tp / (tp + fp) if (tp + fp) else 0.0
    recall_real = tp / (tp + fn) if (tp + fn) else 0.0
    f1_real = (
        2 * precision_real * recall_real / (precision_real + recall_real)
        if (precision_real + recall_real)
        else 0.0
    )

    macro_f1 = (f1_fake + f1_real) / 2.0

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "acc": acc,
        "precision_fake": precision_fake,
        "recall_fake": recall_fake,
        "f1_fake": f1_fake,
        "precision_real": precision_real,
        "recall_real": recall_real,
        "f1_real": f1_real,
        "macro_f1": macro_f1,
    }


def _print_block(title, threshold, m):
    print("\n" + title)
    print("threshold:", round(float(threshold), 3))
    print("accuracy:", round(m["acc"] * 100.0, 2), "%")
    print("macro_f1:", round(m["macro_f1"], 4))
    print("f1_fake:", round(m["f1_fake"], 4), "| f1_real:", round(m["f1_real"], 4))
    print("\nConfusion Matrix (counts):")
    print("            Pred Fake   Pred Real")
    print(f"True Fake   {m['tn']:9d} {m['fp']:10d}")
    print(f"True Real   {m['fn']:9d} {m['tp']:10d}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sanity checks for class mapping, counts, and threshold tuning."
    )
    parser.add_argument(
        "--objective",
        choices=["macro-f1", "real-safe"],
        default="macro-f1",
        help="Optimization objective for tuned threshold.",
    )
    parser.add_argument(
        "--min-real-precision",
        type=float,
        default=0.75,
        help="Used for real-safe objective. Keeps thresholds with precision_real >= this value and chooses best recall_real.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}.")
    if not os.path.isdir(VALID_DIR):
        raise FileNotFoundError(f"Validation directory not found at {VALID_DIR}.")

    print("Loading model from:", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)

    if not hasattr(model, "input_shape") or len(model.input_shape) != 4:
        raise RuntimeError("Unexpected model input shape; expected rank-4 image input.")

    img_size = (int(model.input_shape[1]), int(model.input_shape[2]))
    print("Using validation image size:", img_size)

    val_data = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
        VALID_DIR,
        target_size=img_size,
        batch_size=args.batch_size,
        class_mode="binary",
        classes=["fake", "real"],
        shuffle=False,
    )

    expected = {"fake": 0, "real": 1}
    if val_data.class_indices != expected:
        raise RuntimeError(
            f"Unexpected class mapping: {val_data.class_indices}. Expected: {expected}"
        )

    y_true = val_data.classes.astype(int)
    counts = np.bincount(y_true)

    print("\nClass indices:", val_data.class_indices)
    print("Sample counts -> fake:", int(counts[0]) if len(counts) > 0 else 0, ", real:", int(counts[1]) if len(counts) > 1 else 0)

    probs = model.predict(val_data, verbose=0).ravel()

    baseline_pred = (probs > 0.5).astype(int)
    baseline = _metrics(y_true, baseline_pred)
    _print_block("Baseline @ 0.5", 0.5, baseline)

    thresholds = np.arange(0.1, 0.901, 0.01)

    best_t = 0.5
    best_m = baseline

    if args.objective == "macro-f1":
        for t in thresholds:
            y_pred = (probs > t).astype(int)
            m = _metrics(y_true, y_pred)
            if m["macro_f1"] > best_m["macro_f1"]:
                best_t = float(t)
                best_m = m
    else:
        found = False
        for t in thresholds:
            y_pred = (probs > t).astype(int)
            m = _metrics(y_true, y_pred)
            if m["precision_real"] >= args.min_real_precision:
                if (not found) or (m["recall_real"] > best_m["recall_real"]):
                    found = True
                    best_t = float(t)
                    best_m = m
        if not found:
            print(
                "\nNo threshold satisfied precision_real >=",
                args.min_real_precision,
                "; falling back to macro-f1 objective.",
            )
            best_t = 0.5
            best_m = baseline
            for t in thresholds:
                y_pred = (probs > t).astype(int)
                m = _metrics(y_true, y_pred)
                if m["macro_f1"] > best_m["macro_f1"]:
                    best_t = float(t)
                    best_m = m

    _print_block("Tuned Threshold", best_t, best_m)

    print("\nRecommended threshold for app.py:", round(best_t, 3))
    print("Expected direction: REAL_IF_HIGH_PROB = True")


if __name__ == "__main__":
    main()
