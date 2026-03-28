import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# ======================
# Configuration
# ======================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)

# Expected structure at project root:
# dataset/
#   train/
#     real/
#     fake/
#   valid/
#     real/
#     fake/
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VALID_DIR = os.path.join(DATASET_ROOT, "valid")

if not (os.path.isdir(TRAIN_DIR) and os.path.isdir(VALID_DIR)):
    raise FileNotFoundError(
        "Expected 'dataset/train' and 'dataset/valid' folders at the project root "
        "(each with 'real' and 'fake' subfolders)."
    )

IMG_SIZE = 300  # EfficientNetB3 default size
BATCH_SIZE = 16  # Reduced batch size for B3 to fit in memory
EPOCHS = 35  # Increase total epochs to allow more fine-tuning
SEED = 42

# Save model inside app/models so it stays with the app
MODEL_SAVE_PATH = os.path.join(APP_DIR, "models", "best_model.keras")

# ======================
# Data Preparation (enhanced augmentation)
# ======================

tf.random.set_seed(SEED)
np.random.seed(SEED)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    seed=SEED,
)

# Compute class weights to handle any class imbalance and improve generalization
class_counts = np.bincount(train_data.classes)
total_samples = class_counts.sum()
class_weights = {
    i: total_samples / (2.0 * count)
    for i, count in enumerate(class_counts)
    if count > 0
}
print("Class counts:", class_counts)
print("Class weights:", class_weights)

val_data = val_datagen.flow_from_directory(
    VALID_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False,
)

print("Classes:", train_data.class_indices)

# ======================
# Load EfficientNetB3
# ======================

base_model = EfficientNetB3(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # Start with frozen base

# ======================
# Build Model
# ======================

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),  # Higher LR for initial training of top layers
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ======================
# Callbacks
# ======================

checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# ======================
# Phase 1: Train Top Layers
# ======================

print("\nStarting Phase 1: Training top layers...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,  # Train only top layers for a few epochs
    callbacks=[checkpoint, reduce_lr, early_stop],
    class_weight=class_weights,
)

# ======================
# Phase 2: Fine-tuning (Unfreeze base model)
# ======================

print("\nStarting Phase 2: Fine-tuning base model...")
base_model.trainable = True

# Freeze only the earlier layers and fine-tune deeper layers
# Unfreeze more layers (leave fewer frozen) so the model can adapt better
for layer in base_model.layers[:40]:
    layer.trainable = False

# Recompile with a lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=3e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    initial_epoch=history.epoch[-1] if history.epoch else 15,
    callbacks=[checkpoint, reduce_lr, early_stop],
    class_weight=class_weights,
)

print("\nTraining complete.")
print(f"Best model saved at: {MODEL_SAVE_PATH}")