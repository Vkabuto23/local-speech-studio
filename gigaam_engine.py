import json
import logging
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

from whisper_engine import convert_to_wav


logger = logging.getLogger("local_whisper.gigaam")
BASE_DIR = Path(__file__).resolve().parent
GIGAAM_RUNTIME_DIR = BASE_DIR / "gigaam runtime"
GIGAAM_WORKER = GIGAAM_RUNTIME_DIR / "gigaam_worker.py"


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def gigaam_python() -> Path:
    configured = os.getenv("LOCAL_SPEECH_GIGAAM_PYTHON")
    candidates = [
        Path(configured).expanduser() if configured else None,
        _venv_python(GIGAAM_RUNTIME_DIR / ".venv-gigaam"),
        _venv_python(BASE_DIR / ".venv"),
    ]
    return next((path for path in candidates if path and path.is_file()), candidates[-1])


def gigaam_cache_dir() -> Path:
    return Path(os.getenv("GIGAAM_CACHE_DIR", "~/.cache/gigaam")).expanduser()


def gigaam_model_cached(model_name: str) -> bool:
    cache_dir = gigaam_cache_dir()
    checkpoint = cache_dir / f"{model_name}.ckpt"
    tokenizer = cache_dir / f"{model_name}_tokenizer.model"
    return checkpoint.is_file() and ("e2e" not in model_name or tokenizer.is_file())


def gigaam_available() -> bool:
    return gigaam_python().is_file() and GIGAAM_WORKER.is_file()


def merge_speech_boundaries(
    timestamps: List[Dict[str, int]],
    sample_rate: int = 16000,
    max_segment_seconds: float = 23.0,
    max_merge_gap_seconds: float = 2.0,
) -> List[Dict[str, float]]:
    boundaries: List[Dict[str, float]] = []
    for item in timestamps:
        start = float(item["start"]) / sample_rate
        end = float(item["end"]) / sample_rate
        if not boundaries:
            boundaries.append({"start": start, "end": end})
            continue
        current = boundaries[-1]
        gap = start - current["end"]
        combined_duration = end - current["start"]
        if gap <= max_merge_gap_seconds and combined_duration <= max_segment_seconds:
            current["end"] = end
        else:
            boundaries.append({"start": start, "end": end})
    return [{"start": round(item["start"], 3), "end": round(item["end"], 3)} for item in boundaries]


def _fallback_boundaries(duration: float, max_segment_seconds: float) -> List[Dict[str, float]]:
    return [
        {"start": round(start, 3), "end": round(min(duration, start + max_segment_seconds), 3)}
        for start in [index * max_segment_seconds for index in range(math.ceil(duration / max_segment_seconds))]
    ]


def transcribe_file_with_gigaam(
    file_path: str,
    model_name: str = "v3_e2e_rnnt",
    device: str = "cuda",
    batch_size: int = 4,
    vad_threshold: float = 0.5,
    vad_min_silence_ms: int = 500,
    max_segment_seconds: float = 23.0,
    ffmpeg_path: str = "ffmpeg",
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    if not gigaam_available():
        raise RuntimeError("GigaAM runtime is not installed. Run setup.ps1 or setup_gigaam.ps1 first.")

    def progress(percent: int, message: str) -> None:
        logger.info("Progress: %d%% | %s", percent, message)
        if on_progress:
            on_progress(percent, message)

    with tempfile.TemporaryDirectory(prefix="local_gigaam_") as tmpdir:
        wav_path = Path(tmpdir) / "input.wav"
        boundaries_path = Path(tmpdir) / "boundaries.json"
        output_path = Path(tmpdir) / "result.json"
        progress(5, "Подготавливаю аудио для GigaAM")
        convert_to_wav(file_path, str(wav_path), ffmpeg_path)

        progress(10, "Ищу речевые фрагменты")
        audio = decode_audio(str(wav_path), sampling_rate=16000)
        duration = len(audio) / 16000
        timestamps = get_speech_timestamps(
            audio,
            VadOptions(
                threshold=vad_threshold,
                min_speech_duration_ms=250,
                max_speech_duration_s=max_segment_seconds,
                min_silence_duration_ms=vad_min_silence_ms,
                speech_pad_ms=200,
            ),
            sampling_rate=16000,
        )
        boundaries = merge_speech_boundaries(
            timestamps,
            max_segment_seconds=max_segment_seconds,
        )
        if not boundaries and duration > 0:
            boundaries = _fallback_boundaries(duration, max_segment_seconds)
        boundaries_path.write_text(json.dumps(boundaries), encoding="utf-8")

        if gigaam_model_cached(model_name):
            progress(15, f"Модель GigaAM {model_name} найдена в кэше")
        else:
            progress(15, f"Скачиваю модель GigaAM {model_name}. Первый запуск займёт больше времени")

        cmd = [
            str(gigaam_python()),
            str(GIGAAM_WORKER),
            "--input-wav",
            str(wav_path),
            "--boundaries-json",
            str(boundaries_path),
            "--output-json",
            str(output_path),
            "--model",
            model_name,
            "--device",
            device,
            "--batch-size",
            str(batch_size),
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=str(GIGAAM_RUNTIME_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        output_lines: List[str] = []
        if proc.stdout:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                output_lines.append(line)
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logger.info("GigaAM worker: %s", line)
                    continue
                if event.get("type") == "progress":
                    progress(int(event.get("percent", 0)), str(event.get("message", "GigaAM")))
        code = proc.wait()
        if code != 0 or not output_path.exists():
            tail = "\n".join(output_lines[-20:])
            raise RuntimeError(f"GigaAM worker failed with code {code}: {tail}")

        result = json.loads(output_path.read_text(encoding="utf-8"))
        result["vad_segment_count"] = len(boundaries)
        return result
