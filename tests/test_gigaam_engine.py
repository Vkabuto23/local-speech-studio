import unittest

from gigaam_engine import merge_speech_boundaries


class GigaAMEngineTests(unittest.TestCase):
    def test_nearby_speech_chunks_are_merged_with_original_timestamps(self) -> None:
        timestamps = [
            {"start": 1600, "end": 80000},
            {"start": 88000, "end": 160000},
            {"start": 400000, "end": 480000},
        ]
        result = merge_speech_boundaries(timestamps, max_segment_seconds=23)
        self.assertEqual(result, [{"start": 0.1, "end": 10.0}, {"start": 25.0, "end": 30.0}])

    def test_chunks_are_not_merged_past_gigaam_short_audio_limit(self) -> None:
        timestamps = [
            {"start": 0, "end": 192000},
            {"start": 200000, "end": 384000},
        ]
        result = merge_speech_boundaries(timestamps, max_segment_seconds=23)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
