import unittest

from runtime_settings import profile_for, validate_settings


class RuntimeSettingsTests(unittest.TestCase):
    def test_rtx_5090_balanced_profile_uses_measured_batch_size(self) -> None:
        profile = profile_for(32, "balanced", gpu_available=True)
        self.assertEqual(profile["model"], "large-v3-turbo")
        self.assertEqual(profile["batch_size"], 32)
        self.assertEqual(profile["num_workers"], 1)

    def test_quality_profile_keeps_vram_headroom(self) -> None:
        profile = profile_for(32, "quality", gpu_available=True)
        self.assertEqual(profile["model"], "large-v3")
        self.assertLessEqual(profile["batch_size"], 24)

    def test_cpu_profile_disables_batched_inference(self) -> None:
        profile = profile_for(0, "balanced", gpu_available=False)
        self.assertEqual(profile["device"], "cpu")
        self.assertEqual(profile["compute_type"], "int8")
        self.assertFalse(profile["batched_inference"])

    def test_cpu_float16_is_rejected(self) -> None:
        current = {
            "hardware": {"profile": "manual", "vram_gb": 0},
            "transcriptor": {
                "model": "small",
                "device": "cpu",
                "compute_type": "int8",
                "beam_size": 5,
                "batch_size": 1,
                "cpu_threads": 4,
                "num_workers": 1,
                "device_index": 0,
                "vad_min_silence_ms": 500,
            },
            "diarization": {"default_engine": "nemo", "default_speakers": 4},
            "max_upload_mb": 2048,
        }
        with self.assertRaisesRegex(ValueError, "CPU supports"):
            validate_settings(current, {"transcriptor": {"device": "cpu", "compute_type": "float16"}})

    def test_gigaam_can_be_selected_as_transcription_engine(self) -> None:
        current = {
            "hardware": {"profile": "balanced", "vram_gb": 32},
            "transcriptor": {
                "engine": "whisper",
                "model": "large-v3-turbo",
                "device": "cuda",
                "compute_type": "float16",
                "beam_size": 5,
                "batch_size": 32,
                "cpu_threads": 8,
                "num_workers": 1,
                "device_index": 0,
                "vad_min_silence_ms": 500,
            },
            "gigaam": {},
            "diarization": {"default_engine": "nemo", "default_speakers": 4},
            "max_upload_mb": 2048,
        }
        result = validate_settings(
            current,
            {
                "transcriptor": {"engine": "gigaam"},
                "gigaam": {"model": "v3_e2e_rnnt", "device": "cuda", "batch_size": 4},
            },
        )
        self.assertEqual(result["transcriptor"]["engine"], "gigaam")
        self.assertEqual(result["gigaam"]["model"], "v3_e2e_rnnt")

    def test_unknown_transcription_engine_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Transcription engine"):
            validate_settings(
                {
                    "transcriptor": {},
                    "hardware": {},
                    "gigaam": {},
                    "diarization": {},
                },
                {"transcriptor": {"engine": "unknown"}},
            )

    def test_automatic_speaker_count_is_allowed(self) -> None:
        current = {
            "hardware": {"profile": "balanced", "vram_gb": 8},
            "transcriptor": {},
            "gigaam": {},
            "diarization": {"default_engine": "nemo", "default_speakers": 4},
        }

        result = validate_settings(current, {"diarization": {"default_speakers": 0}})

        self.assertEqual(result["diarization"]["default_speakers"], 0)


if __name__ == "__main__":
    unittest.main()
