import logging
import os
import tempfile
import time
import warnings
import wave
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote\.audio\.core\.io")
warnings.filterwarnings("ignore", message=".*torchcodec is not installed correctly.*", category=UserWarning)

from pyannote.audio import Pipeline

from whisper_engine import convert_to_wav

logger = logging.getLogger("local_whisper.diarization")

_PIPELINE_CACHE: Dict[Tuple[str, str], Pipeline] = {}


def _hf_token() -> Optional[str]:
    return os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")


def resolve_device(preferred: str = "cuda") -> str:
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_pipeline(model_id: str, device: str = "cuda") -> Pipeline:
    token = _hf_token()
    key = (model_id, device)
    if key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[key]

    logger.info("Loading diarization pipeline: %s | device=%s", model_id, device)
    started = time.time()
    pipeline = Pipeline.from_pretrained(model_id, token=token)
    pipeline.to(torch.device(device))
    _PIPELINE_CACHE[key] = pipeline
    logger.info("Diarization pipeline loaded in %.2fs", time.time() - started)
    return pipeline


def load_pcm_wav(path: str) -> Tuple[torch.Tensor, int]:
    with wave.open(path, "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise ValueError("Expected 16-bit PCM WAV after ffmpeg conversion.")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    waveform = torch.from_numpy(audio).unsqueeze(0)
    return waveform, sample_rate


def diarize_file(
    file_path: str,
    model_id: str = "pyannote-community/speaker-diarization-community-1",
    device: str = "cuda",
    ffmpeg_path: str = "ffmpeg",
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> List[Dict[str, Any]]:
    def progress(percent: int, message: str) -> None:
        if on_progress:
            on_progress(percent, message)
        logger.info("Diarization progress: %d%% | %s", percent, message)

    device = resolve_device(device)
    pipeline = get_pipeline(model_id, device)

    with tempfile.TemporaryDirectory(prefix="local_diarization_") as tmpdir:
        wav_path = os.path.join(tmpdir, "input.wav")
        progress(82, "Preparing audio for diarization")
        convert_to_wav(file_path, wav_path, ffmpeg_path)

        waveform, sample_rate = load_pcm_wav(wav_path)
        waveform = waveform.to(torch.device(device))

        kwargs: Dict[str, Any] = {}
        if num_speakers:
            kwargs["num_speakers"] = num_speakers
        if min_speakers:
            kwargs["min_speakers"] = min_speakers
        if max_speakers:
            kwargs["max_speakers"] = max_speakers

        progress(86, "Running speaker diarization")
        diarization_output = pipeline({"waveform": waveform, "sample_rate": sample_rate}, **kwargs)
        diarization = getattr(
            diarization_output,
            "exclusive_speaker_diarization",
            getattr(diarization_output, "speaker_diarization", diarization_output),
        )

    turns: List[Dict[str, Any]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append(
            {
                "start": round(float(turn.start), 3),
                "end": round(float(turn.end), 3),
                "speaker": str(speaker),
            }
        )
    progress(96, f"Diarization complete: {len(turns)} speaker turns")
    return turns


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _speaker_for_interval(start: float, end: float, speaker_turns: List[Dict[str, Any]]) -> Optional[str]:
    if not speaker_turns:
        return None
    scored = [
        (_overlap(start, end, float(turn["start"]), float(turn["end"])), turn)
        for turn in speaker_turns
    ]
    best_overlap, best_turn = max(scored, key=lambda item: item[0])
    if best_overlap > 0:
        return str(best_turn["speaker"])

    midpoint = (start + end) / 2
    nearest = min(
        speaker_turns,
        key=lambda turn: min(
            abs(midpoint - float(turn["start"])),
            abs(midpoint - float(turn["end"])),
        ),
    )
    return str(nearest["speaker"])


def _join_word_tokens(tokens: List[str]) -> str:
    text = ""
    no_space_before = set(".,!?;:%)]}»")
    no_space_after = set("([{«")
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if not text or token[0] in no_space_before or token.startswith("-") or text[-1] in no_space_after:
            text += token
        else:
            text += " " + token
    return text.strip()


def _smooth_short_speaker_islands(words: List[Dict[str, Any]]) -> None:
    """Remove brief A-B-A label jitter without flattening normal speaker turns."""
    for _ in range(2):
        runs: List[tuple[int, int, Optional[str]]] = []
        run_start = 0
        for index in range(1, len(words) + 1):
            if index == len(words) or words[index].get("speaker") != words[run_start].get("speaker"):
                runs.append((run_start, index, words[run_start].get("speaker")))
                run_start = index
        changed = False
        for index in range(1, len(runs) - 1):
            start, end, _ = runs[index]
            previous_speaker = runs[index - 1][2]
            next_speaker = runs[index + 1][2]
            duration = float(words[end - 1].get("end") or 0) - float(words[start].get("start") or 0)
            if previous_speaker == next_speaker and end - start <= 2 and duration <= 1.0:
                for word in words[start:end]:
                    word["speaker"] = previous_speaker
                changed = True
        if not changed:
            break


def _speaker_segments_from_words(words: List[Dict[str, Any]], max_gap: float = 1.2) -> List[Dict[str, Any]]:
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for word in sorted(words, key=lambda item: float(item.get("start") or 0)):
        if not current:
            current = [word]
            continue
        gap = float(word.get("start") or 0) - float(current[-1].get("end") or 0)
        if word.get("speaker") != current[-1].get("speaker") or gap > max_gap:
            groups.append(current)
            current = [word]
        else:
            current.append(word)
    if current:
        groups.append(current)

    segments: List[Dict[str, Any]] = []
    for index, group in enumerate(groups):
        probabilities = [float(item["probability"]) for item in group if item.get("probability") is not None]
        segments.append(
            {
                "id": index,
                "start": round(float(group[0].get("start") or 0), 3),
                "end": round(float(group[-1].get("end") or 0), 3),
                "text": _join_word_tokens([str(item.get("text") or "") for item in group]),
                "speaker": group[0].get("speaker") or "SPEAKER_UNKNOWN",
                "word_count": len(group),
                "avg_word_probability": round(sum(probabilities) / len(probabilities), 4) if probabilities else None,
            }
        )
    return segments


def assign_speakers(result: Dict[str, Any], speaker_turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not speaker_turns:
        result["speaker_turns"] = []
        result["diarized"] = False
        return result

    for word in result.get("word_segments", []):
        start = float(word.get("start") or 0)
        end = float(word.get("end") or start)
        word["speaker"] = _speaker_for_interval(start, end, speaker_turns)

    words = result.get("word_segments", [])
    if words:
        _smooth_short_speaker_islands(words)
        result["whisper_segments"] = [dict(segment) for segment in result.get("segments", [])]
        result["segments"] = _speaker_segments_from_words(words)
    else:
        for segment in result.get("segments", []):
            start = float(segment.get("start") or 0)
            end = float(segment.get("end") or start)
            segment["speaker"] = _speaker_for_interval(start, end, speaker_turns)

    result["speaker_turns"] = sorted(speaker_turns, key=lambda turn: float(turn["start"]))
    result["diarized"] = True
    return result
