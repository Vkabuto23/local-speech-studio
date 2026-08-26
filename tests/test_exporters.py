import unittest

from exporters import _wrap_preformatted_lines, render_export, timestamp


class ExporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.result = {
            "full_text": "Проверка кириллицы: ёжик, Сергей, Ирина.",
            "segments": [
                {
                    "id": 0,
                    "start": 0,
                    "end": 1.9996,
                    "text": "Проверка кириллицы: ёжик, Сергей, Ирина.",
                }
            ],
            "language": "ru",
            "model": "large-v3-turbo",
        }

    def test_timestamp_carries_rounded_milliseconds(self) -> None:
        self.assertEqual(timestamp(1.9996), "00:00:02.000")

    def test_text_exports_use_utf8_bom(self) -> None:
        for format_name in ("txt", "md"):
            payload, _ = render_export(self.result, format_name, "simple")
            self.assertTrue(payload.startswith(b"\xef\xbb\xbf"))
            self.assertIn("Проверка".encode("utf-8"), payload)

    def test_pdf_export_embeds_unicode_font(self) -> None:
        payload, media_type = render_export(self.result, "pdf", "full")
        self.assertEqual(media_type, "application/pdf")
        self.assertTrue(payload.startswith(b"%PDF"))
        self.assertGreater(len(payload), 10_000)

    def test_full_markdown_uses_selected_engine_name(self) -> None:
        self.result["engine"] = "gigaam"
        payload, _ = render_export(self.result, "md", "full")
        self.assertIn("# GigaAM Full Response".encode("utf-8"), payload)

    def test_long_raw_json_lines_wrap_without_losing_characters(self) -> None:
        source = "абв" * 100
        wrapped = _wrap_preformatted_lines(source, width=25)
        self.assertTrue(all(len(line) <= 25 for line in wrapped))
        self.assertEqual("".join(wrapped), source)


if __name__ == "__main__":
    unittest.main()
