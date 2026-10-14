import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
CONFIG_PATH = APP_DIR / "config.py"

spec = importlib.util.spec_from_file_location("app_config", CONFIG_PATH)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)


class ThresholdConfigTests(unittest.TestCase):
    def test_defaults_when_file_missing(self):
        missing_path = os.path.join(tempfile.gettempdir(), "missing-threshold-file.json")
        settings = config.load_threshold_settings(missing_path)
        self.assertFalse(settings["loaded"])
        self.assertEqual(settings["threshold"], config.DEFAULT_THRESHOLD)
        self.assertEqual(settings["real_if_high_prob"], config.DEFAULT_REAL_IF_HIGH_PROB)

    def test_loads_threshold_artifact(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "threshold.json")
            payload = {"threshold": 0.37, "real_if_high_prob": False}
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)

            settings = config.load_threshold_settings(path)
            self.assertTrue(settings["loaded"])
            self.assertAlmostEqual(settings["threshold"], 0.37, places=6)
            self.assertFalse(settings["real_if_high_prob"])


if __name__ == "__main__":
    unittest.main()
