import importlib.util
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
APP_PATH = APP_DIR / "app.py"


def _install_tf_keras_stubs():
    numpy_module = types.ModuleType("numpy")
    numpy_module.expand_dims = lambda arr, axis=0: arr
    numpy_module.float32 = float
    numpy_module.concatenate = lambda values, axis=0: values
    sys.modules["numpy"] = numpy_module

    tf_module = types.ModuleType("tensorflow")
    tf_module.data = types.SimpleNamespace(AUTOTUNE=1)

    def _no_model(*_args, **_kwargs):
        raise FileNotFoundError("no model")

    tf_module.keras = types.SimpleNamespace(
        models=types.SimpleNamespace(load_model=_no_model),
        utils=types.SimpleNamespace(
            image_dataset_from_directory=lambda *a, **k: [],
            load_img=lambda *a, **k: None,
            img_to_array=lambda *a, **k: None,
        ),
    )

    keras_module = types.ModuleType("keras")
    keras_utils_module = types.ModuleType("keras.utils")
    keras_utils_module.load_img = lambda *a, **k: None
    keras_utils_module.img_to_array = lambda *a, **k: None
    keras_apps_module = types.ModuleType("keras.applications")
    keras_effnet_module = types.ModuleType("keras.applications.efficientnet")
    keras_effnet_module.preprocess_input = lambda x: x

    sys.modules["tensorflow"] = tf_module
    sys.modules["keras"] = keras_module
    sys.modules["keras.utils"] = keras_utils_module
    sys.modules["keras.applications"] = keras_apps_module
    sys.modules["keras.applications.efficientnet"] = keras_effnet_module


class FlaskAppSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_tf_keras_stubs()
        sys.path.insert(0, str(APP_DIR))
        spec = importlib.util.spec_from_file_location("fake_image_app", APP_PATH)
        cls.app_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.app_module)
        cls.app_module.app.template_folder = str(APP_DIR / "templates")

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.app_module.app.config.update(TESTING=True, UPLOAD_FOLDER=self.tmp_dir.name)
        self.client = self.app_module.app.test_client()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_index_get_returns_ok(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Upload image", response.data)

    def test_post_upload_without_model_shows_clear_error(self):
        response = self.client.post(
            "/",
            data={"file": (io.BytesIO(b"fake-image-content"), "sample.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Model is not loaded", response.data)

    def test_rejects_unsupported_extension(self):
        response = self.client.post(
            "/",
            data={"file": (io.BytesIO(b"not-image"), "sample.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Unsupported file format", response.data)


if __name__ == "__main__":
    unittest.main()
