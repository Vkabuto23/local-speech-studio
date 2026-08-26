import unittest

from diarization_engine import assign_speakers


class DiarizationAssignmentTests(unittest.TestCase):
    def test_long_whisper_segment_is_split_on_speaker_change(self) -> None:
        result = {
            "segments": [{"id": 0, "start": 0, "end": 4, "text": "Привет Добрый день Как дела Хорошо"}],
            "word_segments": [
                {"start": 0.1, "end": 0.5, "text": "Привет", "probability": 0.9},
                {"start": 0.6, "end": 1.2, "text": "Добрый", "probability": 0.9},
                {"start": 1.3, "end": 1.8, "text": "день", "probability": 0.9},
                {"start": 2.1, "end": 2.5, "text": "Как", "probability": 0.9},
                {"start": 2.6, "end": 3.0, "text": "дела?", "probability": 0.9},
                {"start": 3.1, "end": 3.7, "text": "Хорошо", "probability": 0.9},
            ],
        }
        turns = [
            {"start": 0, "end": 2, "speaker": "SPEAKER_00"},
            {"start": 2, "end": 4, "speaker": "SPEAKER_01"},
        ]

        assigned = assign_speakers(result, turns)

        self.assertEqual(len(assigned["segments"]), 2)
        self.assertEqual(assigned["segments"][0]["speaker"], "SPEAKER_00")
        self.assertEqual(assigned["segments"][1]["speaker"], "SPEAKER_01")
        self.assertEqual(assigned["segments"][1]["text"], "Как дела? Хорошо")
        self.assertEqual(len(assigned["whisper_segments"]), 1)

    def test_word_in_vad_gap_uses_nearest_speaker(self) -> None:
        result = {
            "segments": [],
            "word_segments": [{"start": 2.1, "end": 2.2, "text": "Да"}],
        }
        turns = [
            {"start": 0, "end": 1, "speaker": "SPEAKER_00"},
            {"start": 3, "end": 4, "speaker": "SPEAKER_01"},
        ]

        assigned = assign_speakers(result, turns)

        self.assertEqual(assigned["word_segments"][0]["speaker"], "SPEAKER_01")

    def test_short_label_island_between_same_speaker_is_smoothed(self) -> None:
        result = {
            "segments": [{"id": 0, "start": 0, "end": 2, "text": "Сейчас я продолжу"}],
            "word_segments": [
                {"start": 0.1, "end": 0.5, "text": "Сейчас"},
                {"start": 0.6, "end": 0.8, "text": "я"},
                {"start": 0.9, "end": 1.4, "text": "продолжу"},
            ],
        }
        turns = [
            {"start": 0, "end": 0.55, "speaker": "SPEAKER_00"},
            {"start": 0.55, "end": 0.85, "speaker": "SPEAKER_01"},
            {"start": 0.85, "end": 2, "speaker": "SPEAKER_00"},
        ]

        assigned = assign_speakers(result, turns)

        self.assertEqual(len(assigned["segments"]), 1)
        self.assertEqual(assigned["segments"][0]["speaker"], "SPEAKER_00")


if __name__ == "__main__":
    unittest.main()
