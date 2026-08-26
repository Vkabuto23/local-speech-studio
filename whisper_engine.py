import json
import gc
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.utils import download_model

logger = logging.getLogger("local_whisper.engine")

ModelKey = Tuple[str, str, str, int, int, int]
_MODEL_CACHE: Dict[ModelKey, WhisperModel] = {}
_BATCHED_CACHE: Dict[ModelKey, BatchedInferencePipeline] = {}


def whisper_model_cached(model_name: str) -> bool:
    if Path(model_name).expanduser().exists():
        return True
    try:
        download_model(model_name, local_files_only=True)
        return True
    except Exception:
        return False


def get_model(
    model_name: str,
    device: str,
    compute_type: str,
    cpu_threads: int = 0,
    num_workers: int = 1,
    device_index: int = 0,
) -> WhisperModel:
    key = (model_name, device, compute_type, cpu_threads, num_workers, device_index)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    logger.info("Loading Whisper model: %s | device=%s | compute=%s", model_name, device, compute_type)
    started = time.time()
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
        device_index=device_index,
    )
    _MODEL_CACHE[key] = model
    logger.info("Whisper model loaded in %.2fs", time.time() - started)
    return model


def get_batched_model(
    model_name: str,
    device: str,
    compute_type: str,
    cpu_threads: int = 0,
    num_workers: int = 1,
    device_index: int = 0,
) -> BatchedInferencePipeline:
    key = (model_name, device, compute_type, cpu_threads, num_workers, device_index)
    if key in _BATCHED_CACHE:
        return _BATCHED_CACHE[key]

    model = get_model(model_name, device, compute_type, cpu_threads, num_workers, device_index)
    logger.info("Initializing BatchedInferencePipeline")
    started = time.time()
    batched_model = BatchedInferencePipeline(model=model)
    _BATCHED_CACHE[key] = batched_model
    logger.info("Batched pipeline initialized in %.2fs", time.time() - started)
    return batched_model


def clear_model_cache() -> None:
    """Release cached CTranslate2 models after a runtime configuration change."""
    _BATCHED_CACHE.clear()
    _MODEL_CACHE.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def check_ffmpeg(ffmpeg_path: str = "ffmpeg") -> None:
    try:
        subprocess.run([ffmpeg_path, "-version"], capture_output=True, check=True)
    except Exception as exc:
        raise RuntimeError(f"ffmpeg is not available at '{ffmpeg_path}'. Install ffmpeg or update config.json.") from exc


def _ffprobe_path(ffmpeg_path: str) -> str:
    base = os.path.basename(ffmpeg_path).lower()
    if base.startswith("ffmpeg"):
        return os.path.join(os.path.dirname(ffmpeg_path), os.path.basename(ffmpeg_path).replace("ffmpeg", "ffprobe"))
    return "ffprobe"


def has_audio(file_path: str, ffmpeg_path: str = "ffmpeg") -> bool:
    cmd = [
        _ffprobe_path(ffmpeg_path),
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "json",
        file_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return True
        data = json.loads(proc.stdout or "{}")
        return any(stream.get("codec_type") == "audio" for stream in data.get("streams", []))
    except Exception:
        return True


def convert_to_wav(src: str, dst: str, ffmpeg_path: str = "ffmpeg") -> None:
    if not has_audio(src, ffmpeg_path):
        raise ValueError("Source file has no audio stream.")

    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        src,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        "-c:a",
        "pcm_s16le",
        dst,
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg conversion failed: {err[-1200:]}")


def fmt_hhmmss(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def transcribe_file(
    file_path: str,
    model_name: str = "large-v3-turbo",
    device: str = "cuda",
    compute_type: str = "float16",
    language: Optional[str] = None,
    beam_size: int = 5,
    batched_inference: bool = True,
    batch_size: int = 64,
    cpu_threads: int = 0,
    num_workers: int = 1,
    device_index: int = 0,
    vad_filter: bool = True,
    vad_min_silence_ms: int = 500,
    temperature: float = 0.0,
    ffmpeg_path: str = "ffmpeg",
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    def progress(percent: int, message: str) -> None:
        if on_progress:
            on_progress(percent, message)
        logger.info("Progress: %d%% | %s", percent, message)

    cached = whisper_model_cached(model_name)
    if cached:
        progress(5, f"Модель Whisper {model_name} найдена в кэше")
    else:
        progress(5, f"Скачиваю модель Whisper {model_name}. Первый запуск займёт больше времени")

    model_args = (model_name, device, compute_type, cpu_threads, num_workers, device_index)
    model = get_batched_model(*model_args) if batched_inference else get_model(*model_args)
    progress(9, f"Модель Whisper {model_name} готова")

    with tempfile.TemporaryDirectory(prefix="local_whisper_") as tmpdir:
        wav_path = os.path.join(tmpdir, "input.wav")
        progress(11, "Подготавливаю аудио: 16 кГц, mono WAV")
        convert_to_wav(file_path, wav_path, ffmpeg_path)
        progress(15, "Аудио готово. Запускаю Whisper")

        started = time.time()
        vad_params = {"min_silence_duration_ms": vad_min_silence_ms} if vad_filter else None
        kwargs = {
            "language": language,
            "beam_size": beam_size,
            "vad_filter": vad_filter,
            "vad_parameters": vad_params,
            "temperature": temperature,
            "word_timestamps": True,
        }
        if batched_inference:
            kwargs["batch_size"] = batch_size

        segments_iter, info = model.transcribe(wav_path, **kwargs)
        duration = getattr(info, "duration", None)
        detected_language = getattr(info, "language", None)
        language_probability = getattr(info, "language_probability", None)

        if duration:
            progress(20, f"Длительность аудио: {fmt_hhmmss(duration)}")

        segments: List[Dict[str, Any]] = []
        word_segments: List[Dict[str, Any]] = []
        texts: List[str] = []
        last_pct = 0
        last_tick = time.time()

        for segment in segments_iter:
            text = (getattr(segment, "text", "") or "").strip()
            if text:
                texts.append(text)

            segment_payload = {
                "id": getattr(segment, "id", len(segments)),
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
                "text": text,
                "avg_logprob": getattr(segment, "avg_logprob", None),
                "compression_ratio": getattr(segment, "compression_ratio", None),
                "no_speech_prob": getattr(segment, "no_speech_prob", None),
            }
            segments.append(segment_payload)

            for word in getattr(segment, "words", None) or []:
                word_segments.append(
                    {
                        "start": round(float(word.start), 3),
                        "end": round(float(word.end), 3),
                        "text": (word.word or "").strip(),
                        "probability": getattr(word, "probability", None),
                    }
                )

            if duration and duration > 0:
                pct = int(min(95, max(20, 20 + (float(segment.end) / float(duration)) * 75)))
                now = time.time()
                if pct > last_pct and now - last_tick >= 1.0:
                    progress(pct, f"Transcribing {fmt_hhmmss(segment.end)} / {fmt_hhmmss(duration)}")
                    last_pct = pct
                    last_tick = now

        processing_sec = round(time.time() - started, 2)
        progress(100, f"Done in {processing_sec:.1f}s")

    return {
        "segments": segments,
        "word_segments": word_segments,
        "full_text": " ".join(texts).strip(),
        "language": detected_language,
        "language_probability": language_probability,
        "duration": round(float(duration), 2) if duration else None,
        "processing_sec": processing_sec,
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "batched_inference": batched_inference,
        "batch_size": batch_size if batched_inference else None,
        "cpu_threads": cpu_threads,
        "num_workers": num_workers,
        "device_index": device_index,
    }
