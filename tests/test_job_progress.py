import unittest

from app import JOBS, set_progress


class JobProgressTests(unittest.TestCase):
    def tearDown(self) -> None:
        JOBS.pop("model-download-test", None)

    def test_model_download_message_sets_explicit_phase(self) -> None:
        JOBS["model-download-test"] = {
            "percent": 1,
            "message": "Queued",
            "status": "queued",
        }
        set_progress(
            "model-download-test",
            5,
            "Скачиваю модель GigaAM v3_e2e_rnnt. Первый запуск займёт больше времени",
            "running",
        )
        self.assertEqual(JOBS["model-download-test"]["phase"], "model_download")
        self.assertEqual(JOBS["model-download-test"]["status"], "running")


if __name__ == "__main__":
    unittest.main()
