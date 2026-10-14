import json
import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_THRESHOLD = 0.45
DEFAULT_REAL_IF_HIGH_PROB = True
THRESHOLD_ARTIFACT_PATH = os.path.join(APP_DIR, "models", "threshold.json")
SUPPORTED_EXTENSIONS = (".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp", ".tif", ".tiff")


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def load_threshold_settings(path=THRESHOLD_ARTIFACT_PATH):
    settings = {
        "threshold": float(DEFAULT_THRESHOLD),
        "real_if_high_prob": bool(DEFAULT_REAL_IF_HIGH_PROB),
        "path": os.path.abspath(path),
        "loaded": False,
    }
    if not os.path.exists(path):
        return settings

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return settings

    threshold = payload.get("threshold", payload.get("best_threshold", DEFAULT_THRESHOLD))
    real_if_high_prob = payload.get(
        "real_if_high_prob",
        payload.get("label_direction_real_if_high_prob", DEFAULT_REAL_IF_HIGH_PROB),
    )
    try:
        settings["threshold"] = float(threshold)
    except (TypeError, ValueError):
        settings["threshold"] = float(DEFAULT_THRESHOLD)
    settings["real_if_high_prob"] = _parse_bool(real_if_high_prob)
    settings["loaded"] = True
    return settings
