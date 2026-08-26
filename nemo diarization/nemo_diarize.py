import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from omegaconf import OmegaConf


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def to_wav(input_path: Path, wav_path: Path) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-vn",
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ]
    )


def write_manifest(wav_path: Path, manifest_path: Path, num_speakers: int | None) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    item = {
        "audio_filepath": str(wav_path.resolve()),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": num_speakers,
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    manifest_path.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")


def write_config(manifest_path: Path, out_dir: Path, config_path: Path, num_speakers: int | None) -> None:
    config = {
        "name": "nemo_clustering_diarizer",
        "verbose": True,
        "num_workers": 0,
        "sample_rate": 16000,
        "batch_size": 64,
        "device": "cuda",
        "diarizer": {
            "manifest_filepath": str(manifest_path.resolve()),
            "out_dir": str(out_dir.resolve()),
            "oracle_vad": False,
            "collar": 0.25,
            "ignore_overlap": True,
            "vad": {
                "model_path": "vad_multilingual_marblenet",
                "external_vad_manifest": None,
                "parameters": {
                    "window_length_in_sec": 0.15,
                    "shift_length_in_sec": 0.01,
                    "smoothing": "median",
                    "overlap": 0.875,
                    "onset": 0.5,
                    "offset": 0.3,
                    "pad_onset": 0.2,
                    "pad_offset": 0.2,
                    "min_duration_on": 0.2,
                    "min_duration_off": 0.2,
                    "filter_speech_first": True,
                },
            },
            "speaker_embeddings": {
                "model_path": "titanet_large",
                "parameters": {
                    "window_length_in_sec": [1.5, 1.0, 0.5],
                    "shift_length_in_sec": [0.75, 0.5, 0.25],
                    "multiscale_weights": [0.33, 0.33, 0.34],
                    "save_embeddings": False,
                },
            },
            "clustering": {
                "parameters": {
                    "oracle_num_speakers": num_speakers is not None,
                    "max_num_speakers": num_speakers or 8,
                    "enhanced_count_thres": 80,
                    "max_rp_threshold": 0.25,
                    "sparse_search_volume": 30,
                    "maj_vote_spk_count": False,
                }
            },
            "msdd_model": {"model_path": None, "parameters": {}},
        },
    }
    config_path.write_text(OmegaConf.to_yaml(OmegaConf.create(config)), encoding="utf-8")


def parse_rttm(path: Path) -> list[dict]:
    turns = []
    if not path.exists():
        return turns
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start = float(parts[3])
        duration = float(parts[4])
        turns.append({"start": round(start, 3), "end": round(start + duration, 3), "speaker": parts[7]})
    return turns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--speakers", default="auto")
    parser.add_argument("--work-dir", default="/workspace/nemo_out")
    args = parser.parse_args()

    from nemo.collections.asr.models import ClusteringDiarizer

    input_path = Path(args.input)
    work_dir = Path(args.work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    wav_path = work_dir / "input.wav"
    manifest_path = work_dir / "manifest.json"
    config_path = work_dir / "diarizer.yaml"

    to_wav(input_path, wav_path)
    num_speakers = None if args.speakers == "auto" else int(args.speakers)
    if num_speakers is not None and not 1 <= num_speakers <= 12:
        parser.error("--speakers must be auto or an integer between 1 and 12")
    write_manifest(wav_path, manifest_path, num_speakers)
    write_config(manifest_path, work_dir, config_path, num_speakers)

    cfg = OmegaConf.load(config_path)
    diarizer = ClusteringDiarizer(cfg=cfg)
    diarizer.diarize()

    rttm_files = sorted(work_dir.rglob("*.rttm"))
    print("RTTM files:", [str(path) for path in rttm_files], flush=True)
    if rttm_files:
        turns = parse_rttm(rttm_files[0])
        json_path = work_dir / "speaker_turns.json"
        json_path.write_text(json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"turn_count": len(turns), "json": str(json_path), "rttm": str(rttm_files[0])}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
