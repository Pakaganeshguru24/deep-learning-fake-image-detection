import argparse
import csv
import os
import shutil
import time
from typing import List, Tuple
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, recall_score

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
preprocess_input = tf.keras.applications.efficientnet.preprocess_input

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
MODEL_PATH = os.path.join(APP_DIR, "models", "best_model.keras")
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")
DEFAULT_SOURCE = os.path.join(DATASET_ROOT, "valid")
HARD_REAL_DIR = os.path.join(DATASET_ROOT, "hard_examples", "real")
REPORT_PATH = os.path.join(APP_DIR, "hard_examples_report.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Mine false-positive real images for re-training.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Directory containing real/fake subfolders to scan.")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold. If omitted, macro F1 sweep determines it.")
    parser.add_argument("--top-k", type=int, default=400, help="Maximum number of real hard examples to keep.")
    parser.add_argument("--min-prob", type=float, default=0.55, help="Minimum fake probability required to keep a real sample.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference.")
    return parser.parse_args()


def _macro_f1(y_true, y_pred):
    f1_fake = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_real = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    real_recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    return (f1_real + f1_fake) / 2.0, f1_real, f1_fake, real_recall


def _sweep_threshold(y_true, probs):
    best = {
        "threshold": 0.5,
        "macro_f1": 0.0,
        "real_f1": 0.0,
        "fake_f1": 0.0,
        "real_recall": 0.0,
    }
    for t in np.arange(0.1, 0.9, 0.01):
        y_pred = (probs > t).astype(int)
        macro_f1, real_f1, fake_f1, real_recall = _macro_f1(y_true, y_pred)
        if macro_f1 > best["macro_f1"]:
            best.update({
                "threshold": float(t),
                "macro_f1": macro_f1,
                "real_f1": real_f1,
                "fake_f1": fake_f1,
                "real_recall": real_recall,
            })
    return best


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first.")


def _copy_hard_examples(records: List[Tuple[str, float]]):
    os.makedirs(HARD_REAL_DIR, exist_ok=True)
    copied = []
    timestamp = int(time.time())
    for idx, (src, prob) in enumerate(records):
        ext = os.path.splitext(src)[1].lower() or ".jpg"
        dst_name = f"mined_real_{timestamp}_{idx:05d}_p{prob:.3f}{ext}"
        dst_path = os.path.join(HARD_REAL_DIR, dst_name)
        shutil.copy2(src, dst_path)
        copied.append((src, dst_path, prob))
    return copied


def _write_report(rows):
    headers = ["source_path", "copied_path", "predicted_prob"]
    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def main():
    args = parse_args()
    _ensure_model()
    if not os.path.isdir(args.source):
        raise FileNotFoundError(f"Source directory not found: {args.source}")

    print("Loading model from:", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    img_size = model.input_shape[1] if hasattr(model, "input_shape") else 300

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    data = datagen.flow_from_directory(
        args.source,
        target_size=(img_size, img_size),
        batch_size=args.batch_size,
        class_mode='binary',
        shuffle=False,
    )

    print("Running inference over", len(data.filenames), "images...")
    probs = model.predict(data).ravel()
    y_true = data.classes

    if args.threshold is None:
        sweep = _sweep_threshold(y_true, probs)
        threshold = sweep["threshold"]
        print(
            "Sweep best -> threshold:", round(sweep["threshold"], 3),
            "macro F1:", round(sweep["macro_f1"], 4),
            "real F1:", round(sweep["real_f1"], 4),
            "fake F1:", round(sweep["fake_f1"], 4),
            "real recall:", round(sweep["real_recall"], 4),
        )
    else:
        threshold = args.threshold
        print("Using provided threshold:", threshold)

    filepaths = data.filepaths
    hard_real = []
    for path, label, prob in zip(filepaths, y_true, probs):
        if label == 1 and prob <= threshold and (1.0 - prob) >= args.min_prob:
            hard_real.append((path, float(prob)))

    hard_real.sort(key=lambda item: item[1], reverse=True)
    if args.top_k > 0:
        hard_real = hard_real[:args.top_k]

    if not hard_real:
        print("No hard real samples met the criteria.")
        return

    copied = _copy_hard_examples(hard_real)
    rows = [(src, dst, f"{prob:.4f}") for src, dst, prob in copied]
    _write_report(rows)

    print("Copied", len(copied), "hard real samples to", HARD_REAL_DIR)
    print("Report written to:", REPORT_PATH)


if __name__ == "__main__":
    main()
