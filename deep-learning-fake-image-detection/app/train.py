import argparse
import json
import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

# ======================
# Imports
# ======================

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
EfficientNetB3 = tf.keras.applications.EfficientNetB3
preprocess_input = tf.keras.applications.efficientnet.preprocess_input
layers = tf.keras.layers
Adam = tf.keras.optimizers.Adam
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
EarlyStopping = tf.keras.callbacks.EarlyStopping

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
DEFAULT_MODEL_PATH = os.path.join(APP_DIR, "models", "best_model.keras")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EfficientNet fake-image detector with reproducible settings."
    )
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_DIR,
        help="Dataset root containing train/valid/(optional test) folders.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Where to save best model checkpoint.",
    )
    parser.add_argument("--img-size", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs-phase1", type=int, default=10)
    parser.add_argument("--epochs-phase2", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-metadata-path",
        default=os.path.join(APP_DIR, "models", "last_train_run.json"),
        help="Path to save run metadata JSON.",
    )
    return parser.parse_args()


args = parse_args()

IMG_SIZE = int(args.img_size)
BATCH_SIZE = int(args.batch_size)
EPOCHS_PHASE1 = int(args.epochs_phase1)
EPOCHS_PHASE2 = int(args.epochs_phase2)

DATASET_DIR = os.path.abspath(args.dataset_root)
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")
TEST_DIR = os.path.join(DATASET_DIR, "test")

TRAIN_FAKE_DIR = os.path.join(TRAIN_DIR, "fake")
TRAIN_REAL_DIR = os.path.join(TRAIN_DIR, "real")
VALID_FAKE_DIR = os.path.join(VALID_DIR, "fake")
VALID_REAL_DIR = os.path.join(VALID_DIR, "real")

MODEL_PATH = os.path.abspath(args.model_path)
BAD_FILES_REPORT = os.path.join(APP_DIR, "bad_images_train_report.txt")
RUN_METADATA_PATH = os.path.abspath(args.run_metadata_path)

# Verify dataset exists
if not (os.path.isdir(TRAIN_DIR) and os.path.isdir(VALID_DIR)):
    raise FileNotFoundError(
        "Expected 'dataset/train' and 'dataset/valid' folders at the project root "
        "(each with 'real' and 'fake' subfolders)."
    )

if not (os.path.isdir(TRAIN_FAKE_DIR) and os.path.isdir(TRAIN_REAL_DIR)):
    raise FileNotFoundError(
        "Expected strict binary folders in train split: 'dataset/train/fake' and "
        "'dataset/train/real'."
    )

if not (os.path.isdir(VALID_FAKE_DIR) and os.path.isdir(VALID_REAL_DIR)):
    raise FileNotFoundError(
        "Expected strict binary folders in valid split: 'dataset/valid/fake' and "
        "'dataset/valid/real'."
    )


def scan_unreadable_images(root_dir, class_names=("fake", "real")):
    bad_files = []
    checked = 0
    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for walk_root, _, files in os.walk(class_dir):
            for file_name in files:
                path = os.path.join(walk_root, file_name)
                checked += 1
                try:
                    img = tf.keras.utils.load_img(path)
                    _ = tf.keras.utils.img_to_array(img)
                except Exception:
                    bad_files.append(path)
    return checked, bad_files


print("\nRunning dataset integrity scan (train + valid)...")
checked_train, bad_train = scan_unreadable_images(TRAIN_DIR)
checked_valid, bad_valid = scan_unreadable_images(VALID_DIR)
all_bad = bad_train + bad_valid

print("Checked train files:", checked_train)
print("Checked valid files:", checked_valid)
if all_bad:
    os.makedirs(os.path.dirname(BAD_FILES_REPORT), exist_ok=True)
    with open(BAD_FILES_REPORT, "w", encoding="utf-8") as f:
        f.write("Unreadable image files found before training:\n")
        for path in all_bad:
            f.write(path + "\n")
    raise RuntimeError(
        f"Found {len(all_bad)} unreadable image files. "
        f"See report: {BAD_FILES_REPORT}. Remove/fix them, then retrain."
    )
print("Dataset integrity scan passed: no unreadable files found.")

SEED = int(args.seed)
tf.random.set_seed(SEED)
np.random.seed(SEED)

print("\nTraining config:")
print("Dataset root:", DATASET_DIR)
print("Model path:", MODEL_PATH)
print("Image size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)
print("Epochs phase1:", EPOCHS_PHASE1)
print("Epochs phase2:", EPOCHS_PHASE2)
print("Seed:", SEED)
if os.path.isdir(TEST_DIR):
    print("Test split detected at:", TEST_DIR)


# ======================
# Data Generators (CLEAN)
# ======================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['fake', 'real'],
    shuffle=True,
    seed=SEED
)

val_data = val_datagen.flow_from_directory(
    VALID_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['fake', 'real'],
    shuffle=False
)

# ======================
# Dataset Verification (CRITICAL)
# ======================

print("\n Dataset Verification:")
print("Class indices:", train_data.class_indices)

expected_class_indices = {'fake': 0, 'real': 1}
if train_data.class_indices != expected_class_indices:
    raise RuntimeError(
        f"Unexpected class mapping in train split: {train_data.class_indices}. "
        f"Expected: {expected_class_indices}"
    )
if val_data.class_indices != expected_class_indices:
    raise RuntimeError(
        f"Unexpected class mapping in valid split: {val_data.class_indices}. "
        f"Expected: {expected_class_indices}"
    )

class_counts = np.bincount(train_data.classes)
print("Train fake:", class_counts[0] if len(class_counts) > 0 else 0)
print("Train real:", class_counts[1] if len(class_counts) > 1 else 0)
val_class_counts = np.bincount(val_data.classes)
print("Valid fake:", val_class_counts[0] if len(val_class_counts) > 0 else 0)
print("Valid real:", val_class_counts[1] if len(val_class_counts) > 1 else 0)

# ======================
# Class Weights (BALANCED)
# ======================

class_weights_values = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(class_weights_values))
print("\nClass weights:", class_weights)

# ======================
# Model Architecture
# ======================

base_model = EfficientNetB3(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

# Custom head
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)

x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)

x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)

output = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

# ======================
# Compile (Phase 1)
# ======================

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ======================
# Callbacks
# ======================

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

callbacks = [checkpoint, reduce_lr, early_stop]

# ======================
# Phase 1 Training
# ======================

print("\n🔹 Phase 1: Training top layers...")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_PHASE1,
    callbacks=callbacks,
    class_weight=class_weights
)

# ======================
# Phase 2 Fine-Tuning
# ======================

print("\n🔹 Phase 2: Fine-tuning...")

base_model.trainable = True

for layer in base_model.layers[:-50]:  # freeze most layers
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_PHASE2,
    callbacks=callbacks,
    class_weight=class_weights
)

# ======================
# Done
# ======================

print("\nTraining Complete!")
print("Model saved at:", MODEL_PATH)

os.makedirs(os.path.dirname(RUN_METADATA_PATH), exist_ok=True)
run_metadata = {
    "dataset_name": os.path.basename(os.path.normpath(DATASET_DIR)),
    "dataset_root": DATASET_DIR,
    "train_dir": TRAIN_DIR,
    "valid_dir": VALID_DIR,
    "test_dir_exists": bool(os.path.isdir(TEST_DIR)),
    "model_path": MODEL_PATH,
    "model": "EfficientNetB3",
    "img_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs_phase1": EPOCHS_PHASE1,
    "epochs_phase2": EPOCHS_PHASE2,
    "seed": SEED,
    "class_indices": train_data.class_indices,
    "train_counts": {
        "fake": int(class_counts[0]) if len(class_counts) > 0 else 0,
        "real": int(class_counts[1]) if len(class_counts) > 1 else 0,
    },
    "valid_counts": {
        "fake": int(val_class_counts[0]) if len(val_class_counts) > 0 else 0,
        "real": int(val_class_counts[1]) if len(val_class_counts) > 1 else 0,
    },
}
with open(RUN_METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(run_metadata, f, indent=2)
print("Run metadata saved at:", RUN_METADATA_PATH)