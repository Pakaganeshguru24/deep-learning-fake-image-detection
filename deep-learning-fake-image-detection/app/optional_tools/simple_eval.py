#!/usr/bin/env python
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import numpy as np
    import tensorflow as tf

    ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input

    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(APP_DIR))
    APP_ROOT = os.path.dirname(APP_DIR)
    if APP_ROOT not in sys.path:
        sys.path.insert(0, APP_ROOT)
    from config import load_threshold_settings
    VALID_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'valid')
    MODEL_PATH = os.path.join(APP_ROOT, 'models', 'best_model.keras')
    threshold_settings = load_threshold_settings(os.path.join(APP_ROOT, 'models', 'threshold.json'))
    THRESHOLD = float(threshold_settings["threshold"])
    REAL_IF_HIGH_PROB = bool(threshold_settings["real_if_high_prob"])

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Keep eval size aligned with model input size.
    if hasattr(model, "input_shape") and len(model.input_shape) == 4:
        target_size = (int(model.input_shape[1]), int(model.input_shape[2]))
    else:
        target_size = (300, 300)

    # Prepare validation data
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_data = val_datagen.flow_from_directory(
        VALID_DIR,
        target_size=target_size,
        batch_size=32,
        class_mode='binary',
        shuffle=False,
    )

    # Evaluate
    loss, acc = model.evaluate(val_data, verbose=0)

    # Get predictions
    probs = model.predict(val_data, verbose=0).ravel()
    if REAL_IF_HIGH_PROB:
        preds = (probs > THRESHOLD).astype(int)
    else:
        preds = (probs <= THRESHOLD).astype(int)
    true = val_data.classes

    # Confusion matrix
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(true, preds):
        cm[int(t), int(p)] += 1

    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    report_lines = [
        "=" * 60,
        "MODEL EVALUATION RESULTS",
        "=" * 60,
        "",
        f"Validation Accuracy: {acc * 100:.2f}%",
        f"Validation Loss: {loss:.4f}",
        f"Threshold: {THRESHOLD:.3f}",
        f"real_if_high_prob: {REAL_IF_HIGH_PROB}",
        "",
        "CONFUSION MATRIX:",
        "            Predicted Fake   Predicted Real",
        f"Actual Fake {tn:17d} {fp:16d}",
        f"Actual Real {fn:17d} {tp:16d}",
        "",
        "COUNTS:",
        f"True Negatives:  {tn}",
        f"False Positives: {fp}",
        f"False Negatives: {fn}",
        f"True Positives:  {tp}",
        "",
        "ADDITIONAL METRICS:",
        f"Precision (Real): {precision * 100:.2f}%",
        f"Recall (Real): {recall * 100:.2f}%",
        f"F1-Score: {f1 * 100:.2f}%",
        f"Total Samples: {len(true)}",
    ]
    report = "\n".join(report_lines)

    # Output results
    with open(os.path.join(APP_DIR, 'eval_results.txt'), 'w') as f:
        f.write(report + "\n")

    print(report)

    print("✓ Evaluation complete! Results saved to eval_results.txt")

except Exception as e:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_error.txt'), 'w') as f:
        f.write(f"Error: {str(e)}\n")
        import traceback
        f.write(traceback.format_exc())
    print(f"Error: {e}")
    sys.exit(1)
