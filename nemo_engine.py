import atexit
import json
import logging
import os
import shutil
import subprocess
import time
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("local_whisper.nemo")

BASE_DIR = Path(__file__).resolve().parent
NEMO_DIR = BASE_DIR / "nemo diarization"
NEMO_SCRIPT = NEMO_DIR / "nemo_diarize.py"
NEMO_RUNS_DIR = NEMO_DIR / "runs"
_WORKER: Optional[subprocess.Popen] = None
_WORKER_LOCK = threading.Lock()


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def nemo_python() -> Path:
    configured = os.getenv("LOCAL_SPEECH_NEMO_PYTHON")
    candidates = [
        Path(configured).expanduser() if configured else None,
        _venv_python(NEMO_DIR / ".venv-nemo"),
        _venv_python(BASE_DIR / ".venv"),
    ]
    return next((path for path in candidates if path and path.is_file()), candidates[-1])


def nemo_available() -> bool:
    return nemo_python().exists() and NEMO_SCRIPT.exists()


def _start_worker() -> subprocess.Popen:
    global _WORKER
    if _WORKER and _WORKER.poll() is None:
        return _WORKER
    _WORKER = subprocess.Popen(
        [str(nemo_python()), str(NEMO_SCRIPT), "--server"],
        cwd=str(NEMO_DIR),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert _WORKER.stdout is not None
    for line in _WORKER.stdout:
        logger.info("[NeMo startup] %s", line.rstrip())
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "ready":
            return _WORKER
    raise RuntimeError("NeMo worker stopped during startup")


def warm_nemo_worker() -> None:
    if not nemo_available():
        return
    with _WORKER_LOCK:
        _start_worker()


def shutdown_nemo_worker() -> None:
    global _WORKER
    with _WORKER_LOCK:
        if _WORKER and _WORKER.poll() is None:
            _WORKER.terminate()
            try:
                _WORKER.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _WORKER.kill()
        _WORKER = None


atexit.register(shutdown_nemo_worker)


def diarize_file_with_nemo(
    file_path: str,
    speakers: Optional[int] = None,
    device: str = "cuda",
    batch_size: int = 128,
    num_workers: int = 0,
    vad_segments: Optional[List[Dict[str, Any]]] = None,
    keep_artifacts: bool = False,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> List[Dict[str, Any]]:
    def progress(percent: int, message: str) -> None:
        if on_progress:
            on_progress(percent, message)
        logger.info("NeMo diarization progress: %d%% | %s", percent, message)

    if not nemo_available():
        raise RuntimeError(f"NeMo environment is not ready: {nemo_python()}")

    run_id = f"{int(time.time())}_{Path(file_path).stem[:32]}"
    work_dir = NEMO_RUNS_DIR / run_id
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    vad_segments_path = NEMO_RUNS_DIR / f"{run_id}.vad.json"
    if vad_segments:
        vad_segments_path.write_text(json.dumps(vad_segments, ensure_ascii=False), encoding="utf-8")

    request = {
        "input": str(Path(file_path).resolve()),
        "speakers": str(max(1, int(speakers))) if speakers else "auto",
        "work_dir": str(work_dir.resolve()),
        "device": device,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "vad_segments_json": str(vad_segments_path.resolve()) if vad_segments else None,
    }

    speaker_mode = f"exactly {speakers} speakers" if speakers else "automatic speaker count"
    progress(82, f"Running NeMo diarization with {speaker_mode} on {device}, batch {batch_size}")
    try:
        with _WORKER_LOCK:
            proc = _start_worker()
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            proc.stdin.flush()
            for line in proc.stdout:
                logger.info("[NeMo] %s", line.rstrip())
                lowered = line.lower()
                if "perform streaming frame-level vad" in lowered:
                    progress(86, "NeMo VAD")
                elif "extracting embeddings" in lowered:
                    progress(91, "NeMo speaker embeddings")
                elif "outputs are saved" in lowered:
                    progress(96, "NeMo clustering complete")
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "job_complete":
                    break
                if event.get("type") == "job_error":
                    raise RuntimeError(str(event.get("error") or "NeMo worker failed"))
            else:
                raise RuntimeError("NeMo worker stopped before completing the job")
    finally:
        vad_segments_path.unlink(missing_ok=True)

    json_path = work_dir / "speaker_turns.json"
    if not json_path.exists():
        raise RuntimeError(f"NeMo completed but did not produce {json_path}")

    turns = json.loads(json_path.read_text(encoding="utf-8"))
    progress(97, f"NeMo diarization complete: {len(turns)} turns")
    if not keep_artifacts:
        shutil.rmtree(work_dir, ignore_errors=True)
    return turns
