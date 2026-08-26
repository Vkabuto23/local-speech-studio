import asyncio
import json
import logging
import os
import re
import subprocess
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from diarization_engine import assign_speakers, diarize_file
from exporters import diarized_text, render_export, safe_filename
from gigaam_engine import gigaam_available, transcribe_file_with_gigaam
from nemo_engine import NEMO_RUNS_DIR, diarize_file_with_nemo
from runtime_settings import (
    MODEL_CATALOG,
    all_profiles,
    detect_hardware,
    profile_for,
    save_config,
    validate_settings,
)
from whisper_engine import check_ffmpeg, clear_model_cache, transcribe_file

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
CONFIG_PATH = BASE_DIR / "config.json"
DEFAULT_CONFIG_PATH = BASE_DIR / "config.example.json"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CONVERT_DIR = DATA_DIR / "converted"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CONVERT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("local_whisper")

JOBS: Dict[str, Dict[str, Any]] = {}
CONVERSIONS: Dict[str, Dict[str, Any]] = {}
GPU_WORK_LOCK = asyncio.Lock()
QUEUED_JOB_IDS: list[str] = []
ACTIVE_GPU_JOB_ID: Optional[str] = None

app = FastAPI(title="Local Speech Studio", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        if not DEFAULT_CONFIG_PATH.exists():
            raise FileNotFoundError("config.json and config.example.json are missing")
        shutil.copyfile(DEFAULT_CONFIG_PATH, CONFIG_PATH)
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_runtime(config: Dict[str, Any]) -> Dict[str, Any]:
    tc = config.get("transcriptor", {})
    model = str(tc.get("model") or "large-v3-turbo")
    device = str(tc.get("device") or "cuda").lower()
    compute_type = str(tc.get("compute_type") or "float16")
    batched = bool(tc.get("batched_inference", True))

    if device == "cuda" and not cuda_available():
        logger.warning("CUDA requested but unavailable. Falling back to CPU small/int8.")
        device = "cpu"
        compute_type = "int8"
        batched = False
        if model.startswith("large"):
            model = "small"

    return {
        "model_name": model,
        "device": device,
        "compute_type": compute_type,
        "batched_inference": batched,
        "batch_size": int(tc.get("batch_size", 64)),
        "beam_size": int(tc.get("beam_size", 5)),
        "language": tc.get("language"),
        "vad_filter": bool(tc.get("vad_filter", True)),
        "vad_min_silence_ms": int(tc.get("vad_min_silence_ms", 500)),
        "cpu_threads": int(tc.get("cpu_threads", max(1, (os.cpu_count() or 2) // 2))),
        "num_workers": int(tc.get("num_workers", 1)),
        "device_index": int(tc.get("device_index", 0)),
    }


def fallback_runtime(config: Dict[str, Any]) -> Dict[str, Any]:
    tc = config.get("transcriptor", {})
    return {
        "model_name": "small",
        "device": "cpu",
        "compute_type": "int8",
        "batched_inference": False,
        "batch_size": int(tc.get("batch_size", 16)),
        "beam_size": int(tc.get("beam_size", 5)),
        "language": tc.get("language"),
        "vad_filter": bool(tc.get("vad_filter", True)),
        "vad_min_silence_ms": int(tc.get("vad_min_silence_ms", 500)),
        "cpu_threads": max(1, (os.cpu_count() or 2) - 1),
        "num_workers": 1,
        "device_index": 0,
    }


def resolve_gigaam_runtime(config: Dict[str, Any]) -> Dict[str, Any]:
    gc = config.get("gigaam", {})
    device = str(gc.get("device") or "cuda").lower()
    if device == "cuda" and not cuda_available():
        logger.warning("CUDA requested for GigaAM but unavailable. Falling back to CPU.")
        device = "cpu"
    return {
        "model_name": str(gc.get("model") or "v3_e2e_rnnt"),
        "device": device,
        "batch_size": int(gc.get("batch_size", 4)),
        "vad_threshold": float(gc.get("vad_threshold", 0.5)),
        "vad_min_silence_ms": int(gc.get("vad_min_silence_ms", 500)),
        "max_segment_seconds": float(gc.get("max_segment_seconds", 23.0)),
    }


def should_retry_on_cpu(exc: Exception) -> bool:
    message = str(exc).lower()
    gpu_markers = (
        "cuda",
        "cublas",
        "cudnn",
        "out of memory",
        "driver",
        "device index",
        "compute type",
    )
    return any(marker in message for marker in gpu_markers)


def storage_stats() -> Dict[str, Any]:
    roots = (UPLOAD_DIR, CONVERT_DIR, NEMO_RUNS_DIR)
    files = [path for root in roots if root.exists() for path in root.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "total_gb": round(total_bytes / (1024**3), 2),
    }


def set_progress(job_id: str, percent: int, message: str, status: Optional[str] = None) -> None:
    job = JOBS[job_id]
    job["percent"] = percent
    job["message"] = message
    if status:
        job["status"] = status
    lowered = message.lower()
    if "скачиваю модель" in lowered:
        job["phase"] = "model_download"
    elif "модель" in lowered and ("загружаю" in lowered or "найдена" in lowered or "готова" in lowered):
        job["phase"] = "model_load"
    elif status == "done":
        job["phase"] = "done"
    elif status == "error":
        job["phase"] = "error"
    else:
        job["phase"] = "processing"
    job["updated_at"] = time.time()


def set_conversion_progress(job_id: str, percent: int, message: str, status: Optional[str] = None) -> None:
    job = CONVERSIONS[job_id]
    job["percent"] = percent
    job["message"] = message
    if status:
        job["status"] = status
    job["updated_at"] = time.time()


def ffprobe_duration(file_path: str, ffmpeg_path: str = "ffmpeg") -> Optional[float]:
    ffmpeg_file = Path(ffmpeg_path)
    if ffmpeg_file.name.lower().startswith("ffmpeg"):
        ffprobe_name = ffmpeg_file.name.replace("ffmpeg", "ffprobe", 1).replace("FFMPEG", "ffprobe", 1)
        ffprobe_path = str(ffmpeg_file.with_name(ffprobe_name))
    else:
        ffprobe_path = "ffprobe"
    proc = subprocess.run(
        [
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            file_path,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        return float((proc.stdout or "").strip())
    except ValueError:
        return None


def parse_ffmpeg_time(line: str) -> Optional[float]:
    match = re.search(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", line)
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def conversion_command(ffmpeg_path: str, source: str, output: str, audio_format: str) -> list[str]:
    base = [ffmpeg_path, "-y", "-i", source, "-vn"]
    if audio_format == "mp3":
        return base + ["-ac", "2", "-ar", "44100", "-b:a", "192k", output]
    if audio_format == "wav":
        return base + ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", output]
    if audio_format == "m4a":
        return base + ["-ac", "2", "-ar", "44100", "-c:a", "aac", "-b:a", "192k", output]
    raise ValueError("format must be mp3, wav, or m4a")


def conversion_media_type(audio_format: str) -> str:
    return {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "m4a": "audio/mp4",
    }[audio_format]


def run_conversion(job_id: str, file_path: str, audio_format: str) -> None:
    config = load_config()
    ffmpeg_path = config.get("ffmpeg_path", "ffmpeg")
    source_name = CONVERSIONS[job_id]["filename"]
    output_name = f"{job_id}.{audio_format}"
    output_path = CONVERT_DIR / output_name

    try:
        duration = ffprobe_duration(file_path, ffmpeg_path)
        set_conversion_progress(job_id, 5, "Preparing conversion", "running")
        cmd = conversion_command(ffmpeg_path, file_path, str(output_path), audio_format)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        last_percent = 5
        if proc.stderr:
            for line in proc.stderr:
                current_time = parse_ffmpeg_time(line)
                if duration and current_time is not None and duration > 0:
                    percent = int(min(95, max(5, 5 + (current_time / duration) * 90)))
                    if percent > last_percent:
                        last_percent = percent
                        set_conversion_progress(job_id, percent, "Converting video to audio", "running")

        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg conversion failed with code {code}")

        CONVERSIONS[job_id]["output_path"] = str(output_path)
        CONVERSIONS[job_id]["audio_format"] = audio_format
        CONVERSIONS[job_id]["download_name"] = f"{safe_filename(Path(source_name).stem)}.{audio_format}"
        set_conversion_progress(job_id, 100, "Complete", "done")
    except Exception as exc:
        logger.exception("Conversion %s failed", job_id)
        CONVERSIONS[job_id]["error"] = str(exc)
        set_conversion_progress(job_id, -1, f"Error: {exc}", "error")
    finally:
        Path(file_path).unlink(missing_ok=True)


async def run_transcription(
    job_id: str,
    file_path: str,
    language: Optional[str],
    diarize: bool = False,
    diarization_engine: str = "nemo",
    speakers: Optional[int] = None,
) -> None:
    global ACTIVE_GPU_JOB_ID
    config = load_config()
    engine = str(config.get("transcriptor", {}).get("engine", "whisper")).lower()
    runtime = resolve_runtime(config) if engine == "whisper" else resolve_gigaam_runtime(config)
    if language and engine == "whisper":
        runtime["language"] = language
    ffmpeg_path = config.get("ffmpeg_path", "ffmpeg")

    def on_progress(percent: int, message: str) -> None:
        if diarize:
            percent = int(min(80, max(1, percent * 0.8)))
        set_progress(job_id, percent, message, "running")

    try:
        async with GPU_WORK_LOCK:
            ACTIVE_GPU_JOB_ID = job_id
            if job_id in QUEUED_JOB_IDS:
                QUEUED_JOB_IDS.remove(job_id)
            set_progress(job_id, 3, "GPU slot acquired", "running")
            if engine == "gigaam":
                result = await asyncio.to_thread(
                    transcribe_file_with_gigaam,
                    file_path=file_path,
                    ffmpeg_path=ffmpeg_path,
                    on_progress=on_progress,
                    **runtime,
                )
            else:
                try:
                    result = await asyncio.to_thread(
                        transcribe_file,
                        file_path=file_path,
                        ffmpeg_path=ffmpeg_path,
                        on_progress=on_progress,
                        **runtime,
                    )
                except Exception as primary_error:
                    if not should_retry_on_cpu(primary_error):
                        raise
                    logger.warning("GPU runtime failed for %s: %s", job_id, primary_error)
                    fallback = fallback_runtime(config)
                    if language:
                        fallback["language"] = language
                    set_progress(job_id, 12, "Primary runtime failed. Retrying with CPU fallback", "running")
                    result = await asyncio.to_thread(
                        transcribe_file,
                        file_path=file_path,
                        ffmpeg_path=ffmpeg_path,
                        on_progress=on_progress,
                        **fallback,
                    )
                result["engine"] = "whisper"

            if diarize:
                dc = config.get("diarization", {})
                set_progress(job_id, 81, "Starting speaker diarization", "running")
                if diarization_engine == "nemo":
                    speaker_turns = await asyncio.to_thread(
                        diarize_file_with_nemo,
                        file_path=file_path,
                        speakers=speakers,
                        keep_artifacts=bool(dc.get("keep_artifacts", False)),
                        on_progress=lambda percent, message: set_progress(job_id, percent, message, "running"),
                    )
                else:
                    speaker_turns = await asyncio.to_thread(
                        diarize_file,
                        file_path=file_path,
                        model_id=dc.get("model", "pyannote-community/speaker-diarization-community-1"),
                        device=dc.get("device", "cuda"),
                        ffmpeg_path=ffmpeg_path,
                        num_speakers=speakers,
                        on_progress=lambda percent, message: set_progress(job_id, percent, message, "running"),
                    )
                result = assign_speakers(result, speaker_turns)
                result["diarization_engine"] = diarization_engine
                result["speaker_count_requested"] = speakers
                detected_speakers = {
                    str(segment.get("speaker"))
                    for segment in result.get("segments", [])
                    if segment.get("speaker") is not None
                }
                result["speaker_count_detected"] = len(
                    detected_speakers
                    or {str(turn.get("speaker")) for turn in speaker_turns if turn.get("speaker") is not None}
                )
            else:
                result["diarized"] = False

        JOBS[job_id]["result"] = result
        set_progress(job_id, 100, "Complete", "done")
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        JOBS[job_id]["error"] = str(exc)
        set_progress(job_id, -1, f"Error: {exc}", "error")
    finally:
        if job_id in QUEUED_JOB_IDS:
            QUEUED_JOB_IDS.remove(job_id)
        if ACTIVE_GPU_JOB_ID == job_id:
            ACTIVE_GPU_JOB_ID = None
        Path(file_path).unlink(missing_ok=True)


@app.on_event("startup")
def startup() -> None:
    config = load_config()
    try:
        check_ffmpeg(config.get("ffmpeg_path", "ffmpeg"))
    except Exception as exc:
        logger.warning("ffmpeg check failed at startup: %s", exc)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    config = load_config()
    engine = config.get("transcriptor", {}).get("engine", "whisper")
    return {
        "status": "ok",
        "service": "local-whisper",
        "engine": engine,
        "model": config.get("gigaam", {}).get("model") if engine == "gigaam" else config.get("transcriptor", {}).get("model"),
        "gigaam_available": gigaam_available(),
        "gpu_busy": GPU_WORK_LOCK.locked(),
        "queued_jobs": len(QUEUED_JOB_IDS),
    }


def settings_payload() -> Dict[str, Any]:
    config = load_config()
    hardware = detect_hardware()
    configured_vram = float(config.get("hardware", {}).get("vram_gb") or hardware["vram_gb"])
    return {
        "config": config,
        "hardware": hardware,
        "profiles": all_profiles(configured_vram, bool(hardware["gpu_available"])),
        "model_catalog": MODEL_CATALOG,
        "gigaam_available": gigaam_available(),
        "storage": storage_stats(),
        "runtime": {
            "gpu_busy": GPU_WORK_LOCK.locked(),
            "active_job_id": ACTIVE_GPU_JOB_ID,
            "queued_jobs": len(QUEUED_JOB_IDS),
        },
    }


@app.get("/api/settings")
def get_settings() -> Dict[str, Any]:
    return settings_payload()


@app.get("/api/settings/recommend")
def recommend_settings(vram_gb: float, profile: str = "balanced") -> Dict[str, Any]:
    hardware = detect_hardware()
    if profile not in {"balanced", "quality", "speed"}:
        raise HTTPException(400, "profile must be balanced, quality, or speed")
    if not 0 <= vram_gb <= 256:
        raise HTTPException(400, "vram_gb must be between 0 and 256")
    return profile_for(vram_gb, profile, bool(hardware["gpu_available"]))


@app.put("/api/settings")
async def update_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    if GPU_WORK_LOCK.locked():
        raise HTTPException(409, "Wait for the active GPU job before changing runtime settings")
    try:
        config = validate_settings(load_config(), payload)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    await asyncio.to_thread(save_config, CONFIG_PATH, config)
    await asyncio.to_thread(clear_model_cache)
    return settings_payload()


@app.post("/api/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
    diarize: bool = Form(default=False),
    diarization_engine: str = Form(default="nemo"),
    speakers: int = Form(default=0),
) -> Dict[str, Any]:
    diarization_engine = diarization_engine.lower()
    if diarization_engine not in {"nemo", "pyannote"}:
        raise HTTPException(400, "diarization_engine must be nemo or pyannote")
    if not 0 <= speakers <= 12:
        raise HTTPException(400, "speakers must be 0 (auto) or between 1 and 12")
    requested_speakers = speakers or None
    config = load_config()
    max_upload_mb = int(config.get("max_upload_mb", 2048))
    job_id = str(uuid.uuid4())
    source_name = file.filename or "media"
    ext = Path(source_name).suffix or ".bin"
    dest = UPLOAD_DIR / f"{job_id}{ext}"
    total = 0

    try:
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_upload_mb * 1024 * 1024:
                    raise HTTPException(413, f"File is larger than {max_upload_mb} MB")
                out.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    JOBS[job_id] = {
        "id": job_id,
        "filename": source_name,
        "path": str(dest),
        "status": "queued",
        "percent": 1,
        "message": "Queued",
        "phase": "queued",
        "created_at": time.time(),
        "updated_at": time.time(),
        "result": None,
        "error": None,
        "diarize": diarize,
        "diarization_engine": diarization_engine if diarize else None,
        "speakers": requested_speakers if diarize else None,
    }
    QUEUED_JOB_IDS.append(job_id)
    background_tasks.add_task(
        run_transcription,
        job_id,
        str(dest),
        language,
        diarize,
        diarization_engine,
        requested_speakers,
    )
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    queue_position = QUEUED_JOB_IDS.index(job_id) + 1 if job_id in QUEUED_JOB_IDS else None
    message = job["message"]
    if queue_position:
        message = f"Queued: position {queue_position}"
    result = job.get("result") or {}
    return {
        "id": job["id"],
        "filename": job["filename"],
        "status": job["status"],
        "percent": job["percent"],
        "message": message,
        "phase": job.get("phase", "processing"),
        "queue_position": queue_position,
        "error": job["error"],
        "has_result": job.get("result") is not None,
        "diarize": job.get("diarize", False),
        "diarization_engine": job.get("diarization_engine"),
        "speakers": job.get("speakers"),
        "speakers_detected": result.get("speaker_count_detected"),
        "text_preview": diarized_text(result),
    }


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str, format: str = "txt", view: str = "simple") -> Response:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.get("status") != "done" or not job.get("result"):
        raise HTTPException(409, "Job is not complete")

    try:
        payload, media_type = render_export(job["result"], format, view)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    basename = safe_filename(Path(job["filename"]).stem)
    filename = f"{basename}.{view}.{format.lower()}"
    quoted_filename = quote(filename)
    return Response(
        content=payload,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quoted_filename}"},
    )


@app.post("/api/convert")
async def create_conversion(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    audio_format: str = Form(default="mp3"),
) -> Dict[str, Any]:
    audio_format = audio_format.lower()
    if audio_format not in {"mp3", "wav", "m4a"}:
        raise HTTPException(400, "audio_format must be mp3, wav, or m4a")

    config = load_config()
    max_upload_mb = int(config.get("max_upload_mb", 2048))
    job_id = str(uuid.uuid4())
    source_name = file.filename or "video"
    ext = Path(source_name).suffix or ".bin"
    dest = UPLOAD_DIR / f"{job_id}.convert{ext}"
    total = 0

    try:
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_upload_mb * 1024 * 1024:
                    raise HTTPException(413, f"File is larger than {max_upload_mb} MB")
                out.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    CONVERSIONS[job_id] = {
        "id": job_id,
        "filename": source_name,
        "path": str(dest),
        "status": "queued",
        "percent": 1,
        "message": "Queued",
        "created_at": time.time(),
        "updated_at": time.time(),
        "audio_format": audio_format,
        "output_path": None,
        "download_name": None,
        "error": None,
    }
    background_tasks.add_task(run_conversion, job_id, str(dest), audio_format)
    return {"job_id": job_id}


@app.get("/api/convert/{job_id}")
def get_conversion(job_id: str) -> Dict[str, Any]:
    job = CONVERSIONS.get(job_id)
    if not job:
        raise HTTPException(404, "Conversion not found")
    return {
        "id": job["id"],
        "filename": job["filename"],
        "status": job["status"],
        "percent": job["percent"],
        "message": job["message"],
        "audio_format": job["audio_format"],
        "error": job["error"],
        "has_result": bool(job.get("output_path")),
    }


@app.get("/api/convert/{job_id}/download")
def download_conversion(job_id: str) -> Response:
    job = CONVERSIONS.get(job_id)
    if not job:
        raise HTTPException(404, "Conversion not found")
    if job.get("status") != "done" or not job.get("output_path"):
        raise HTTPException(409, "Conversion is not complete")

    output_path = Path(job["output_path"])
    if not output_path.exists():
        raise HTTPException(404, "Converted file was removed")

    filename = job.get("download_name") or output_path.name
    return FileResponse(
        path=str(output_path),
        media_type=conversion_media_type(job["audio_format"]),
        filename=filename,
    )


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> Dict[str, str]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.get("status") in {"queued", "running"}:
        raise HTTPException(409, "A queued or running job cannot be deleted")
    JOBS.pop(job_id, None)
    path = job.get("path")
    if path and os.path.exists(path):
        os.remove(path)
    return {"status": "deleted"}


@app.delete("/api/convert/{job_id}")
def delete_conversion(job_id: str) -> Dict[str, str]:
    job = CONVERSIONS.get(job_id)
    if not job:
        raise HTTPException(404, "Conversion not found")
    if job.get("status") in {"queued", "running"}:
        raise HTTPException(409, "A queued or running conversion cannot be deleted")
    CONVERSIONS.pop(job_id, None)
    for key in ("path", "output_path"):
        path = job.get(key)
        if path and os.path.exists(path):
            os.remove(path)
    return {"status": "deleted"}


if __name__ == "__main__":
    import uvicorn

    cfg = load_config().get("server", {})
    uvicorn.run(
        "app:app",
        host=cfg.get("host", "127.0.0.1"),
        port=int(cfg.get("port", 8015)),
        reload=False,
    )
