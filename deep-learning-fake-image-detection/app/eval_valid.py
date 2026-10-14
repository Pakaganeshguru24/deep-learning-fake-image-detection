import os
import argparse
import time
import numpy as np
import tensorflow as tf
from keras.applications.efficientnet import preprocess_input
from config import (
    THRESHOLD_ARTIFACT_PATH,
    load_threshold_settings,
    SUPPORTED_EXTENSIONS,
)

# Match paths and settings from train.py
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VALID_DIR = os.path.join(DATASET_ROOT, "valid")
TEST_DIR = os.path.join(DATASET_ROOT, "test")

MODEL_PATH = os.path.join(APP_DIR, "models", "best_model.keras")
SUPPORTED_EXTS = SUPPORTED_EXTENSIONS


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trained model on a real/fake dataset directory. "
            "Can evaluate train, valid, test, all, or a custom --data-dir."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=DATASET_ROOT,
        help="Dataset root containing train/valid/test subfolders.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to dataset folder containing 'fake' and 'real' subfolders.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test", "all"],
        default="valid",
        help="Dataset split under dataset/ to evaluate when --data-dir is not provided.",
    )
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help="Path to trained .keras model file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold for converting probabilities to labels (overrides config/artifact).",
    )
    parser.add_argument(
        "--real-if-high-prob",
        action="store_true",
        default=None,
        help="Interpret high model probability as class 'real'.",
    )
    parser.add_argument(
        "--real-if-low-prob",
        action="store_true",
        default=False,
        help="Interpret low model probability as class 'real'.",
    )
    parser.add_argument(
        "--threshold-config",
        default=THRESHOLD_ARTIFACT_PATH,
        help="Path to threshold JSON artifact produced by tune_threshold.py.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help=(
            "Continue evaluation even when some files are unreadable. "
            "By default, unreadable files raise an error."
        ),
    )
    parser.add_argument(
        "--bad-files-report",
        default=os.path.join(APP_DIR, "bad_images_report.txt"),
        help="Path to write unreadable image file paths.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print progress every N processed images.",
    )
    return parser.parse_args()


def collect_candidate_files(class_dir):
    paths = []
    unsupported = []
    for root, _, files in os.walk(class_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in SUPPORTED_EXTS:
                paths.append(os.path.join(root, name))
            else:
                unsupported.append(os.path.join(root, name))
    paths.sort()
    unsupported.sort()
    return paths, unsupported


def evaluate_directory(
    model,
    eval_dir,
    split_name,
    img_size,
    threshold,
    real_if_high_prob,
    batch_size,
    allow_mismatch,
    bad_files_report,
    progress_every,
):
    fake_dir = os.path.join(eval_dir, "fake")
    real_dir = os.path.join(eval_dir, "real")
    if not (os.path.isdir(fake_dir) and os.path.isdir(real_dir)):
        raise FileNotFoundError(
            f"Expected '{eval_dir}' to contain 'fake' and 'real' subfolders."
        )

    fake_candidates, fake_unsupported = collect_candidate_files(fake_dir)
    real_candidates, real_unsupported = collect_candidate_files(real_dir)
    candidate_count = len(fake_candidates) + len(real_candidates)
    unsupported_total = len(fake_unsupported) + len(real_unsupported)

    print(f"\n=== Evaluating split: {split_name} ===")
    print("Dataset path:", eval_dir)
    print("File counts (candidate image extensions) -> fake:", len(fake_candidates), "real:", len(real_candidates))
    if unsupported_total > 0:
        print(
            "Warning: Found",
            unsupported_total,
            "unsupported files in class folders (ignored by TensorFlow).",
        )

    all_paths = fake_candidates + real_candidates
    all_labels = [0] * len(fake_candidates) + [1] * len(real_candidates)
    total_files = len(all_paths)

    good_true = []
    good_probs = []
    bad_paths = []

    batch_imgs = []
    batch_labels = []
    start_time = time.perf_counter()
    processed_files = 0

    def _fmt_duration(seconds):
        seconds = max(0, int(seconds))
        hrs = seconds // 3600
        mins = (seconds % 3600) // 60
        secs = seconds % 60
        if hrs > 0:
            return f"{hrs:02d}:{mins:02d}:{secs:02d}"
        return f"{mins:02d}:{secs:02d}"

    def _print_progress(force=False):
        if total_files == 0:
            return
        if (not force) and (processed_files % progress_every != 0):
            return
        elapsed = time.perf_counter() - start_time
        rate = processed_files / elapsed if elapsed > 0 else 0.0
        remaining = total_files - processed_files
        eta = (remaining / rate) if rate > 0 else 0.0
        pct = (processed_files / total_files) * 100.0
        print(
            f"Progress [{split_name}]: {processed_files}/{total_files} "
            f"({pct:.1f}%) | loaded={len(good_true)} bad={len(bad_paths)} "
            f"| {rate:.2f} img/s | elapsed={_fmt_duration(elapsed)} "
            f"ETA={_fmt_duration(eta)}"
        )

    def flush_batch():
        if not batch_imgs:
            return
        x = np.stack(batch_imgs, axis=0).astype(np.float32)
        x = preprocess_input(x)
        probs_batch = model.predict(x, verbose=0).ravel()
        good_probs.extend(probs_batch.tolist())
        good_true.extend(batch_labels)
        batch_imgs.clear()
        batch_labels.clear()

    for path, label in zip(all_paths, all_labels):
        try:
            img = tf.keras.utils.load_img(path, target_size=img_size)
            arr = tf.keras.utils.img_to_array(img)
            batch_imgs.append(arr)
            batch_labels.append(label)
            if len(batch_imgs) >= batch_size:
                flush_batch()
        except Exception:
            bad_paths.append(path)
        finally:
            processed_files += 1
            _print_progress(force=(processed_files == total_files))

    flush_batch()
    _print_progress(force=True)

    if bad_paths:
        os.makedirs(os.path.dirname(bad_files_report), exist_ok=True)
        with open(bad_files_report, "a", encoding="utf-8") as f:
            f.write(f"\n[{split_name}] unreadable images in {eval_dir}\n")
            for p in bad_paths:
                f.write(p + "\n")

    loaded_count = len(good_true)
    print("Decodable images loaded:", loaded_count)
    if bad_paths:
        print(
            "Warning: Found",
            len(bad_paths),
            "unreadable image files. Paths appended to:",
            bad_files_report,
        )

    if loaded_count != candidate_count:
        msg = (
            f"Decodable image count mismatch for '{split_name}'. "
            f"Candidate image files: {candidate_count}, decodable: {loaded_count}."
        )
        if allow_mismatch:
            print("Warning:", msg)
        else:
            raise RuntimeError(msg)

    if loaded_count == 0:
        raise RuntimeError(f"No decodable images found in '{eval_dir}'.")

    true = np.array(good_true, dtype=np.int32)
    probs = np.array(good_probs, dtype=np.float32)
    if real_if_high_prob:
        preds = (probs > threshold).astype(np.int32)
    else:
        preds = (probs <= threshold).astype(np.int32)

    bce = tf.keras.losses.binary_crossentropy(
        tf.convert_to_tensor(true, dtype=tf.float32),
        tf.convert_to_tensor(probs, dtype=tf.float32),
    )
    loss = float(tf.reduce_mean(bce).numpy())
    acc = float((preds == true).mean())

    print("Loss:", loss)
    print("Accuracy:", acc * 100.0, "%")

    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(true, preds):
        row = 0 if int(t) == 0 else 1
        col = 0 if int(p) == 0 else 1
        cm[row, col] += 1

    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    print("\nConfusion Matrix (counts):")
    print("            Pred Fake   Pred Real")
    print(f"True Fake   {tn:9d} {fp:10d}")
    print(f"True Real   {fn:9d} {tp:10d}")

    acc_manual = (preds == true).mean() * 100.0
    print("Manual accuracy check:", acc_manual, "%")

    return {
        "split": split_name,
        "loss": float(loss),
        "accuracy": float(acc),
        "loaded_count": loaded_count,
        "expected_supported": candidate_count,
        "bad_count": len(bad_paths),
    }


args = parse_args()
model_path = os.path.abspath(args.model_path)
dataset_root = os.path.abspath(args.dataset_root)
train_dir = os.path.join(dataset_root, "train")
valid_dir = os.path.join(dataset_root, "valid")
test_dir = os.path.join(dataset_root, "test")
if args.real_if_low_prob:
    direction_override = False
elif args.real_if_high_prob is True:
    direction_override = True
else:
    direction_override = None

threshold_settings = load_threshold_settings(os.path.abspath(args.threshold_config))
if args.threshold is None:
    threshold = float(threshold_settings["threshold"])
else:
    threshold = float(args.threshold)
if direction_override is None:
    real_if_high_prob = bool(threshold_settings["real_if_high_prob"])
else:
    real_if_high_prob = bool(direction_override)
batch_size = int(args.batch_size)
allow_mismatch = bool(args.allow_mismatch)
bad_files_report = os.path.abspath(args.bad_files_report)
progress_every = max(1, int(args.progress_every))

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Could not find trained model at '{model_path}'. Run train.py first."
    )

if args.data_dir:
    eval_targets = [("custom", os.path.abspath(args.data_dir))]
else:
    if args.split == "train":
        eval_targets = [("train", train_dir)]
    elif args.split == "valid":
        eval_targets = [("valid", valid_dir)]
    elif args.split == "test":
        eval_targets = [("test", test_dir)]
    else:
        eval_targets = [("train", train_dir), ("valid", valid_dir), ("test", test_dir)]

for split_name, split_dir in eval_targets:
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Expected dataset split folder at '{split_dir}'.")

print("Loading model from:", model_path)
print("Using threshold:", threshold)
print("Using real_if_high_prob:", real_if_high_prob)
print("Using batch size:", batch_size)
print("Dataset root:", dataset_root)
print("Threshold config path:", os.path.abspath(args.threshold_config))
print("Progress update every:", progress_every, "images")
print("Bad files report path:", bad_files_report)
model = tf.keras.models.load_model(model_path)

# Derive expected input size from model
if hasattr(model, "input_shape") and len(model.input_shape) == 4:
    img_size = (model.input_shape[1], model.input_shape[2])
else:
    img_size = (300, 300)

print("Using validation image size:", img_size)

results = []
for split_name, split_dir in eval_targets:
    metrics = evaluate_directory(
        model=model,
        eval_dir=split_dir,
        split_name=split_name,
        img_size=img_size,
        threshold=threshold,
        real_if_high_prob=real_if_high_prob,
        batch_size=batch_size,
        allow_mismatch=allow_mismatch,
        bad_files_report=bad_files_report,
        progress_every=progress_every,
    )
    results.append(metrics)

if len(results) > 1:
    print("\n=== Summary ===")
    for item in results:
        print(
            f"{item['split']}: accuracy={item['accuracy'] * 100.0:.2f}% "
            f"loss={item['loss']:.4f} "
            f"loaded={item['loaded_count']} "
            f"expected_supported={item['expected_supported']}"
        )
