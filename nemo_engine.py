import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("local_whisper.nemo")

BASE_DIR = Path(__file__).resolve().parent
NEMO_DIR = BASE_DIR / "nemo diarization"
NEMO_SCRIPT = NEMO_DIR / "nemo_diarize_test.py"
NEMO_RUNS_DIR = NEMO_DIR / "runs"


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


def diarize_file_with_nemo(
    file_path: str,
    speakers: int = 4,
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

    cmd = [
        str(nemo_python()),
        str(NEMO_SCRIPT),
        "--input",
        str(Path(file_path).resolve()),
        "--speakers",
        str(max(1, int(speakers or 4))),
        "--work-dir",
        str(work_dir.resolve()),
    ]

    progress(82, f"Running NeMo diarization with {speakers} speakers")
    proc = subprocess.Popen(
        cmd,
        cwd=str(NEMO_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        logger.info("[NeMo] %s", line.rstrip())
        lowered = line.lower()
        if "perform streaming frame-level vad" in lowered:
            progress(86, "NeMo VAD")
        elif "extracting embeddings" in lowered:
            progress(91, "NeMo speaker embeddings")
        elif "outputs are saved" in lowered:
            progress(96, "NeMo clustering complete")

    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"NeMo diarization failed with code {code}")

    json_path = work_dir / "speaker_turns.json"
    if not json_path.exists():
        raise RuntimeError(f"NeMo completed but did not produce {json_path}")

    turns = json.loads(json_path.read_text(encoding="utf-8"))
    progress(97, f"NeMo diarization complete: {len(turns)} turns")
    if not keep_artifacts:
        shutil.rmtree(work_dir, ignore_errors=True)
    return turns
