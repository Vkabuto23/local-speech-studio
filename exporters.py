import io
import json
import os
import re
from html import escape
from typing import Any, Dict, Iterable, List, Tuple

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, XPreformatted


def safe_filename(name: str) -> str:
    clean = re.sub(r"[^\w\-. а-яА-ЯёЁ]+", "_", name, flags=re.UNICODE).strip("._ ")
    return clean or "transcript"


def timestamp(seconds: float) -> str:
    total_millis = max(0, round(float(seconds or 0) * 1000))
    total_seconds, millis = divmod(total_millis, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _metadata_lines(result: Dict[str, Any]) -> List[str]:
    fields = [
        ("Engine", result.get("engine")),
        ("Language", result.get("language")),
        ("Language probability", result.get("language_probability")),
        ("Duration", result.get("duration")),
        ("Processing seconds", result.get("processing_sec")),
        ("Model", result.get("model")),
        ("Device", result.get("device")),
        ("Compute type", result.get("compute_type")),
        ("Batched inference", result.get("batched_inference")),
        ("Batch size", result.get("batch_size")),
        ("Diarized", result.get("diarized")),
    ]
    return [f"{key}: {value}" for key, value in fields if value is not None]


def diarized_text(result: Dict[str, Any]) -> str:
    segments = result.get("segments", [])
    if not result.get("diarized") or not segments:
        return (result.get("full_text") or "").strip()

    lines: List[str] = []
    current_speaker = None
    buffer: List[str] = []
    for segment in segments:
        speaker = segment.get("speaker") or "SPEAKER_UNKNOWN"
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        if current_speaker is None:
            current_speaker = speaker
        if speaker != current_speaker:
            lines.append(f"{current_speaker}: {' '.join(buffer).strip()}")
            lines.append("")
            current_speaker = speaker
            buffer = []
        buffer.append(text)
    if buffer and current_speaker:
        lines.append(f"{current_speaker}: {' '.join(buffer).strip()}")
    return "\n".join(lines).strip()


def render_txt(result: Dict[str, Any], view: str) -> bytes:
    if view == "simple":
        content = diarized_text(result) + "\n"
    else:
        content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    return content.encode("utf-8-sig")


def render_md(result: Dict[str, Any], view: str) -> bytes:
    if view == "simple":
        content = "# Transcript\n\n" + diarized_text(result) + "\n"
    else:
        engine_name = "GigaAM" if result.get("engine") == "gigaam" else "Whisper"
        lines = [f"# {engine_name} Full Response", "", "## Metadata", ""]
        lines.extend(f"- {line}" for line in _metadata_lines(result))
        lines.extend(["", "## Segments", ""])
        for item in result.get("segments", []):
            speaker = f" **{item.get('speaker')}**" if item.get("speaker") else ""
            lines.append(f"- `{timestamp(item.get('start', 0))} - {timestamp(item.get('end', 0))}`{speaker} {item.get('text', '').strip()}")
        lines.extend(["", "## Raw JSON", "", "```json", json.dumps(result, ensure_ascii=False, indent=2), "```", ""])
        content = "\n".join(lines)
    return content.encode("utf-8-sig")


def _font_candidates() -> Iterable[Tuple[str, str]]:
    windir = os.environ.get("WINDIR", r"C:\Windows")
    yield "ArialUnicode", os.path.join(windir, "Fonts", "arial.ttf")
    yield "SegoeUI", os.path.join(windir, "Fonts", "segoeui.ttf")
    yield "DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _register_unicode_font() -> str:
    for font_name, font_path in _font_candidates():
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont(font_name, font_path))
                return font_name
            except Exception:
                continue
    return "Helvetica"


def _paragraph(text: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(text or "")).replace("\n", "<br/>"), style)


def _on_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(doc.font_name, 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawRightString(A4[0] - 16 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _chunk_segments(
    segments: List[Dict[str, Any]],
    max_gap: float = 2.5,
    max_duration: float = 90.0,
) -> List[List[Dict[str, Any]]]:
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    prev_end = None
    for seg in segments:
        start = float(seg.get("start") or 0)
        first_start = float(current[0].get("start") or 0) if current else start
        speaker_changed = bool(
            current
            and current[-1].get("speaker")
            and seg.get("speaker")
            and current[-1].get("speaker") != seg.get("speaker")
        )
        if current and (
            (prev_end is not None and start - prev_end > max_gap)
            or start - first_start > max_duration
            or speaker_changed
        ):
            groups.append(current)
            current = []
        current.append(seg)
        prev_end = float(seg.get("end") or start)
    if current:
        groups.append(current)
    return groups


def _simple_pdf_paragraphs(result: Dict[str, Any]) -> List[str]:
    if result.get("diarized"):
        return [item.strip() for item in re.split(r"\n{2,}", diarized_text(result)) if item.strip()]

    segments = result.get("segments", [])
    if segments:
        paragraphs = []
        for group in _chunk_segments(segments, max_gap=4.0, max_duration=90.0):
            text = " ".join((item.get("text") or "").strip() for item in group).strip()
            if text:
                paragraphs.append(text)
        if paragraphs:
            return paragraphs
    return [item.strip() for item in re.split(r"\n{2,}", diarized_text(result)) if item.strip()]


def _wrap_preformatted_lines(text: str, width: int = 105) -> List[str]:
    wrapped: List[str] = []
    for line in text.splitlines():
        if not line:
            wrapped.append("")
            continue
        wrapped.extend(line[index : index + width] for index in range(0, len(line), width))
    return wrapped


def render_pdf(result: Dict[str, Any], view: str) -> bytes:
    font_name = _register_unicode_font()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=16 * mm,
    )
    doc.font_name = font_name

    base = getSampleStyleSheet()
    title = ParagraphStyle("TitleLocal", parent=base["Title"], fontName=font_name, fontSize=18, leading=22, spaceAfter=8)
    heading = ParagraphStyle("HeadingLocal", parent=base["Heading2"], fontName=font_name, fontSize=12, leading=15, spaceBefore=8, spaceAfter=4)
    body = ParagraphStyle("BodyLocal", parent=base["BodyText"], fontName=font_name, fontSize=10, leading=14, alignment=TA_LEFT)
    small = ParagraphStyle("SmallLocal", parent=body, fontSize=8, leading=10, textColor=colors.HexColor("#475467"))
    code = ParagraphStyle("CodeLocal", parent=small, fontSize=6.5, leading=8, textColor=colors.HexColor("#344054"))

    story: List[Any] = [_paragraph("Transcript", title)]

    if view == "simple":
        paragraphs = _simple_pdf_paragraphs(result)
        for index, paragraph in enumerate(paragraphs):
            story.append(_paragraph(paragraph, body))
            if index < len(paragraphs) - 1:
                story.append(Spacer(1, 4))
    else:
        story.append(_paragraph("Metadata", heading))
        metadata_rows = [[_paragraph(line.split(": ", 1)[0], small), _paragraph(line.split(": ", 1)[1], small)] for line in _metadata_lines(result)]
        if metadata_rows:
            table = Table(metadata_rows, colWidths=[48 * mm, 112 * mm], hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("FONTNAME", (0, 0), (-1, -1), font_name),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D0D5DD")),
                        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F2F4F7")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 8))

        story.append(_paragraph("Segments", heading))
        for group_index, group in enumerate(_chunk_segments(result.get("segments", [])), start=1):
            group_start = timestamp(group[0].get("start", 0))
            group_end = timestamp(group[-1].get("end", 0))
            story.append(_paragraph(f"Block {group_index}: {group_start} - {group_end}", small))
            rows = []
            for seg in group:
                speaker = f"{seg.get('speaker')}\n" if seg.get("speaker") else ""
                rows.append(
                    [
                        _paragraph(f"{timestamp(seg.get('start', 0))}\n-\n{timestamp(seg.get('end', 0))}", small),
                        _paragraph(f"{speaker}{seg.get('text', '')}", body),
                    ]
                )
            table = Table(rows, colWidths=[35 * mm, 125 * mm], hAlign="LEFT", repeatRows=0)
            table.setStyle(
                TableStyle(
                    [
                        ("FONTNAME", (0, 0), (-1, -1), font_name),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#EAECF0")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 6))

        story.append(PageBreak())
        engine_name = "GigaAM" if result.get("engine") == "gigaam" else "Whisper"
        story.append(_paragraph(f"Raw {engine_name} Response", heading))
        raw = json.dumps(result, ensure_ascii=False, indent=2)
        chunk: List[str] = []
        chunk_size = 0
        for line in _wrap_preformatted_lines(raw):
            if chunk and chunk_size + len(line) > 7000:
                story.append(XPreformatted(escape("\n".join(chunk)), code))
                chunk = []
                chunk_size = 0
            chunk.append(line)
            chunk_size += len(line) + 1
        if chunk:
            story.append(XPreformatted(escape("\n".join(chunk)), code))

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return buffer.getvalue()


def render_export(result: Dict[str, Any], fmt: str, view: str) -> Tuple[bytes, str]:
    fmt = fmt.lower()
    view = view.lower()
    if view not in {"simple", "full"}:
        raise ValueError("view must be simple or full")
    if fmt == "txt":
        return render_txt(result, view), "text/plain; charset=utf-8"
    if fmt == "md":
        return render_md(result, view), "text/markdown; charset=utf-8"
    if fmt == "pdf":
        return render_pdf(result, view), "application/pdf"
    raise ValueError("format must be pdf, md, or txt")
