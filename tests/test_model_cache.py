import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gigaam_engine import gigaam_model_cached
from whisper_engine import whisper_model_cached


class ModelCacheTests(unittest.TestCase):
    def test_local_whisper_model_directory_is_cached(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(whisper_model_cached(directory))

    @patch("whisper_engine.download_model", side_effect=RuntimeError("not cached"))
    def test_missing_whisper_model_is_not_cached(self, _download) -> None:
        self.assertFalse(whisper_model_cached("missing-test-model"))

    def test_gigaam_e2e_requires_checkpoint_and_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            checkpoint = cache / "v3_e2e_rnnt.ckpt"
            tokenizer = cache / "v3_e2e_rnnt_tokenizer.model"
            checkpoint.touch()
            with patch.dict(os.environ, {"GIGAAM_CACHE_DIR": directory}):
                self.assertFalse(gigaam_model_cached("v3_e2e_rnnt"))
                tokenizer.touch()
                self.assertTrue(gigaam_model_cached("v3_e2e_rnnt"))


if __name__ == "__main__":
    unittest.main()
