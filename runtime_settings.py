import json
import os
import subprocess
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


MODEL_CATALOG = [
    {
        "id": "large-v3",
        "label": "Whisper large-v3",
        "quality": "maximum",
        "recommended_vram_gb": 12,
    },
    {
        "id": "large-v3-turbo",
        "label": "Whisper large-v3-turbo",
        "quality": "balanced",
        "recommended_vram_gb": 8,
    },
    {
        "id": "medium",
        "label": "Whisper medium",
        "quality": "compact",
        "recommended_vram_gb": 5,
    },
    {
        "id": "small",
        "label": "Whisper small",
        "quality": "light",
        "recommended_vram_gb": 3,
    },
    {
        "id": "base",
        "label": "Whisper base",
        "quality": "minimum",
        "recommended_vram_gb": 2,
    },
]

_CONFIG_LOCK = threading.Lock()


def detect_hardware() -> Dict[str, Any]:
    hardware: Dict[str, Any] = {
        "gpu_available": False,
        "gpu_name": None,
        "vram_gb": 0.0,
        "cpu_logical_cores": os.cpu_count() or 1,
        "cuda_version": None,
    }
    try:
        import torch

        hardware["gpu_available"] = bool(torch.cuda.is_available())
        hardware["cuda_version"] = torch.version.cuda
        if hardware["gpu_available"]:
            props = torch.cuda.get_device_properties(0)
            hardware["gpu_name"] = props.name
            hardware["vram_gb"] = round(props.total_memory / (1024**3), 1)
            return hardware
    except Exception:
        pass

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        name, memory_mib = output.splitlines()[0].rsplit(",", 1)
        hardware.update(
            gpu_available=True,
            gpu_name=name.strip(),
            vram_gb=round(float(memory_mib.strip()) / 1024, 1),
        )
    except Exception:
        pass
    return hardware


def _batch_for_vram(vram_gb: float) -> int:
    if vram_gb < 4:
        return 4
    if vram_gb < 8:
        return 8
    if vram_gb < 16:
        return 16
    if vram_gb < 24:
        return 24
    return 32


def profile_for(vram_gb: float, profile: str, gpu_available: bool = True) -> Dict[str, Any]:
    logical_cores = os.cpu_count() or 1
    cpu_threads = max(1, logical_cores // 2)
    if not gpu_available or vram_gb < 2:
        return {
            "model": "small",
            "device": "cpu",
            "compute_type": "int8",
            "beam_size": 5,
            "batched_inference": False,
            "batch_size": 1,
            "cpu_threads": max(1, logical_cores - 1),
            "num_workers": 1,
            "device_index": 0,
        }

    batch_size = _batch_for_vram(vram_gb)
    compute_type = "float16" if vram_gb >= 8 else "int8_float16"

    if profile == "quality":
        if vram_gb >= 12:
            model = "large-v3"
            batch_size = min(batch_size, 24)
        elif vram_gb >= 6:
            model = "large-v3-turbo"
        else:
            model = "medium"
        beam_size = 5
    elif profile == "speed":
        model = "large-v3-turbo" if vram_gb >= 8 else "medium"
        batch_size = min(batch_size, 16)
        beam_size = 1
    else:
        model = "large-v3-turbo" if vram_gb >= 6 else "medium"
        beam_size = 5

    return {
        "model": model,
        "device": "cuda",
        "compute_type": compute_type,
        "beam_size": beam_size,
        "batched_inference": True,
        "batch_size": batch_size,
        "cpu_threads": cpu_threads,
        "num_workers": 1,
        "device_index": 0,
    }


def all_profiles(vram_gb: float, gpu_available: bool) -> Dict[str, Dict[str, Any]]:
    return {
        name: profile_for(vram_gb, name, gpu_available)
        for name in ("balanced", "quality", "speed")
    }


def validate_settings(current: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    updated = deepcopy(current)
    hardware = updated.setdefault("hardware", {})
    transcriptor = updated.setdefault("transcriptor", {})
    gigaam = updated.setdefault("gigaam", {})
    diarization = updated.setdefault("diarization", {})

    incoming_hardware = payload.get("hardware", {})
    profile = str(incoming_hardware.get("profile", hardware.get("profile", "balanced")))
    if profile not in {"balanced", "quality", "speed", "manual"}:
        raise ValueError("Unknown hardware profile")
    vram_gb = float(incoming_hardware.get("vram_gb", hardware.get("vram_gb", 0)))
    if not 0 <= vram_gb <= 256:
        raise ValueError("VRAM must be between 0 and 256 GB")
    hardware.update(profile=profile, vram_gb=round(vram_gb, 1))

    incoming_tc = payload.get("transcriptor", {})
    engine = str(incoming_tc.get("engine", transcriptor.get("engine", "whisper"))).lower()
    if engine not in {"whisper", "gigaam"}:
        raise ValueError("Transcription engine must be whisper or gigaam")
    allowed_models = {item["id"] for item in MODEL_CATALOG}
    model = str(incoming_tc.get("model", transcriptor.get("model", "large-v3-turbo")))
    if model not in allowed_models:
        raise ValueError("Unsupported Whisper model")
    device = str(incoming_tc.get("device", transcriptor.get("device", "cuda"))).lower()
    if device not in {"cuda", "cpu"}:
        raise ValueError("Device must be cuda or cpu")
    compute_type = str(incoming_tc.get("compute_type", transcriptor.get("compute_type", "float16")))
    if compute_type not in {"float16", "int8_float16", "int8", "float32"}:
        raise ValueError("Unsupported compute type")
    if device == "cpu" and compute_type not in {"int8", "float32"}:
        raise ValueError("CPU supports int8 or float32 in this service")

    integer_fields = {
        "beam_size": (1, 10),
        "batch_size": (1, 128),
        "cpu_threads": (1, 128),
        "num_workers": (1, 8),
        "device_index": (0, 16),
        "vad_min_silence_ms": (100, 5000),
    }
    transcriptor.update(engine=engine, model=model, device=device, compute_type=compute_type)
    for field, (minimum, maximum) in integer_fields.items():
        value = int(incoming_tc.get(field, transcriptor.get(field, minimum)))
        if not minimum <= value <= maximum:
            raise ValueError(f"{field} must be between {minimum} and {maximum}")
        transcriptor[field] = value
    for field in ("batched_inference", "vad_filter"):
        if field in incoming_tc:
            transcriptor[field] = bool(incoming_tc[field])

    incoming_gigaam = payload.get("gigaam", {})
    gigaam_model = str(incoming_gigaam.get("model", gigaam.get("model", "v3_e2e_rnnt")))
    if gigaam_model not in {"v3_e2e_rnnt", "v3_e2e_ctc"}:
        raise ValueError("Unsupported GigaAM model")
    gigaam_device = str(incoming_gigaam.get("device", gigaam.get("device", "cuda"))).lower()
    if gigaam_device not in {"cuda", "cpu"}:
        raise ValueError("GigaAM device must be cuda or cpu")
    gigaam_batch_size = int(incoming_gigaam.get("batch_size", gigaam.get("batch_size", 4)))
    if not 1 <= gigaam_batch_size <= 32:
        raise ValueError("GigaAM batch_size must be between 1 and 32")
    gigaam_vad_threshold = float(incoming_gigaam.get("vad_threshold", gigaam.get("vad_threshold", 0.5)))
    if not 0.1 <= gigaam_vad_threshold <= 0.9:
        raise ValueError("GigaAM vad_threshold must be between 0.1 and 0.9")
    gigaam_vad_silence = int(
        incoming_gigaam.get("vad_min_silence_ms", gigaam.get("vad_min_silence_ms", 500))
    )
    if not 100 <= gigaam_vad_silence <= 5000:
        raise ValueError("GigaAM vad_min_silence_ms must be between 100 and 5000")
    gigaam_max_segment = float(
        incoming_gigaam.get("max_segment_seconds", gigaam.get("max_segment_seconds", 23.0))
    )
    if not 5 <= gigaam_max_segment <= 24:
        raise ValueError("GigaAM max_segment_seconds must be between 5 and 24")
    gigaam.update(
        model=gigaam_model,
        device=gigaam_device,
        batch_size=gigaam_batch_size,
        vad_threshold=round(gigaam_vad_threshold, 2),
        vad_min_silence_ms=gigaam_vad_silence,
        max_segment_seconds=round(gigaam_max_segment, 1),
    )

    incoming_dc = payload.get("diarization", {})
    engine = str(incoming_dc.get("default_engine", diarization.get("default_engine", "nemo")))
    if engine not in {"nemo", "pyannote"}:
        raise ValueError("Unknown diarization engine")
    speakers = int(incoming_dc.get("default_speakers", diarization.get("default_speakers", 0)))
    if not 0 <= speakers <= 12:
        raise ValueError("Speaker count must be 0 (auto) or between 1 and 12")
    diarization.update(default_engine=engine, default_speakers=speakers)

    max_upload_mb = int(payload.get("max_upload_mb", updated.get("max_upload_mb", 2048)))
    if not 32 <= max_upload_mb <= 16384:
        raise ValueError("Upload limit must be between 32 and 16384 MB")
    updated["max_upload_mb"] = max_upload_mb
    return updated


def save_config(path: Path, config: Dict[str, Any]) -> None:
    serialized = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    temp_path = path.with_suffix(".json.tmp")
    with _CONFIG_LOCK:
        temp_path.write_text(serialized, encoding="utf-8")
        temp_path.replace(path)
