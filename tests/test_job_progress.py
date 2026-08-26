import unittest

from app import JOBS, get_job, set_progress


class JobProgressTests(unittest.TestCase):
    def tearDown(self) -> None:
        JOBS.pop("model-download-test", None)
        JOBS.pop("long-preview-test", None)

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

    def test_completed_job_returns_the_entire_transcript(self) -> None:
        transcript = "Длинная транскрипция " * 500
        JOBS["long-preview-test"] = {
            "id": "long-preview-test",
            "filename": "meeting.webm",
            "status": "done",
            "percent": 100,
            "message": "Complete",
            "phase": "done",
            "error": None,
            "diarize": False,
            "diarization_engine": None,
            "speakers": None,
            "result": {"full_text": transcript, "diarized": False},
        }

        payload = get_job("long-preview-test")

        self.assertEqual(payload["text_preview"], transcript.strip())
        self.assertGreater(len(payload["text_preview"]), 1200)
        self.assertFalse(payload["diarize"])
        self.assertIsNone(payload["diarization_engine"])


if __name__ == "__main__":
    unittest.main()
