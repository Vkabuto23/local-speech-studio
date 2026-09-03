import argparse
import gc
import glob
import json
import os
import shutil
import subprocess
import traceback
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf


def install_fast_vad_smoothing(torch_module) -> None:
    from nemo.collections.asr.models import clustering_diarizer

    def smooth_directory(
        frame_pred_dir: str,
        smoothing_method: str,
        overlap: float,
        window_length_in_sec: float,
        shift_length_in_sec: float,
        num_workers: int,
        out_dir: str | None = None,
    ) -> str:
        if smoothing_method != "median":
            from nemo.collections.asr.parts.utils.vad_utils import generate_overlap_vad_seq

            return generate_overlap_vad_seq(
                frame_pred_dir,
                smoothing_method,
                overlap,
                window_length_in_sec,
                shift_length_in_sec,
                num_workers,
                out_dir,
            )

        output_dir = out_dir or os.path.join(
            frame_pred_dir,
            f"overlap_smoothing_output_{smoothing_method}_{overlap}",
        )
        os.makedirs(output_dir, exist_ok=True)
        frame_len = 0.01
        shift = int(shift_length_in_sec / frame_len)
        segment_frames = int(window_length_in_sec / frame_len + 1)
        target_jump = int(segment_frames * (1 - overlap))
        source_jump = int(target_jump / shift)
        if source_jump < 1:
            raise ValueError("Invalid VAD smoothing overlap: source jump is below one frame")

        start_step = source_jump * shift
        max_overlap = (segment_frames + start_step - 1) // start_step
        for frame_path in glob.glob(os.path.join(frame_pred_dir, "*.frame")):
            values = [float(line) for line in Path(frame_path).read_text(encoding="utf-8").splitlines()]
            frame = torch_module.tensor(values, dtype=torch_module.float32)
            selected = frame[::source_jump]
            target_index = torch_module.arange(len(frame) * shift, dtype=torch_module.long)
            latest_source = torch_module.div(target_index, start_step, rounding_mode="floor")
            offsets = torch_module.arange(max_overlap, dtype=torch_module.long)
            source_index = latest_source[:, None] - offsets[None, :]
            source_start = source_index * start_step
            valid = (source_index >= 0) & (source_index < len(selected))
            valid &= (target_index[:, None] - source_start) < segment_frames
            gathered = selected[source_index.clamp(0, max(0, len(selected) - 1))]
            gathered = gathered.masked_fill(~valid, float("nan"))
            predictions = torch_module.nanquantile(gathered, q=0.5, dim=1)
            nan_mask = torch_module.isnan(predictions)
            if nan_mask.any():
                predictions[nan_mask] = predictions[~nan_mask][-1]

            output_path = Path(output_dir) / f"{Path(frame_path).stem}.{smoothing_method}"
            output_path.write_text(
                "".join(f"{value:.4f}\n" for value in predictions.tolist()),
                encoding="utf-8",
            )
        return output_dir

    clustering_diarizer.generate_overlap_vad_seq = smooth_directory


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


def write_external_vad_manifest(wav_path: Path, source_path: Path, manifest_path: Path) -> None:
    raw = source_path.read_text(encoding="utf-8-sig")
    try:
        payload = json.loads(raw)
        items = payload if isinstance(payload, list) else [payload]
    except json.JSONDecodeError:
        items = [json.loads(line) for line in raw.splitlines() if line.strip()]

    normalized = []
    for item in items:
        try:
            start = max(0.0, float(item.get("start", item.get("offset", 0))))
            end = float(item.get("end", start + float(item.get("duration", 0))))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        normalized.append(
            {
                "audio_filepath": str(wav_path.resolve()),
                "offset": round(start, 3),
                "duration": round(end - start, 3),
                "label": "UNK",
                "uniq_id": wav_path.stem,
            }
        )
    if not normalized:
        raise ValueError("External VAD contains no valid speech segments")
    manifest_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in normalized),
        encoding="utf-8",
    )


def write_config(
    manifest_path: Path,
    out_dir: Path,
    config_path: Path,
    num_speakers: int | None,
    device: str,
    batch_size: int,
    num_workers: int,
    external_vad_manifest: Path | None,
) -> None:
    config = {
        "name": "nemo_clustering_diarizer",
        "verbose": True,
        "num_workers": num_workers,
        "sample_rate": 16000,
        "batch_size": batch_size,
        "device": device,
        "diarizer": {
            "manifest_filepath": str(manifest_path.resolve()),
            "out_dir": str(out_dir.resolve()),
            "oracle_vad": False,
            "collar": 0.25,
            "ignore_overlap": True,
            "vad": {
                "model_path": None if external_vad_manifest else "vad_multilingual_marblenet",
                "external_vad_manifest": str(external_vad_manifest.resolve()) if external_vad_manifest else None,
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


def validate_args(args, parser: argparse.ArgumentParser | None = None) -> None:
    def fail(message: str) -> None:
        if parser:
            parser.error(message)
        raise ValueError(message)

    if not args.input:
        fail("--input is required")
    if not 1 <= args.batch_size <= 1024:
        fail("--batch-size must be between 1 and 1024")
    if not 0 <= args.num_workers <= 16:
        fail("--num-workers must be between 0 and 16")
    if args.speakers != "auto" and not 1 <= int(args.speakers) <= 12:
        fail("--speakers must be auto or an integer between 1 and 12")


def configure_torch(torch_module, device: str) -> None:
    if device != "cuda":
        return
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch_module.backends.cuda.matmul.allow_tf32 = True
    torch_module.backends.cudnn.allow_tf32 = True
    torch_module.backends.cudnn.benchmark = True
    torch_module.set_float32_matmul_precision("high")


def run_job(args, torch_module, diarizer_class) -> None:
    validate_args(args)
    configure_torch(torch_module, args.device)

    input_path = Path(args.input)
    work_dir = Path(args.work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    wav_path = work_dir / "input.wav"
    manifest_path = work_dir / "manifest.json"
    external_vad_manifest = work_dir / "external_vad.json" if args.vad_segments_json else None
    config_path = work_dir / "diarizer.yaml"

    to_wav(input_path, wav_path)
    num_speakers = None if args.speakers == "auto" else int(args.speakers)
    write_manifest(wav_path, manifest_path, num_speakers)
    if external_vad_manifest:
        write_external_vad_manifest(wav_path, Path(args.vad_segments_json), external_vad_manifest)
    write_config(
        manifest_path,
        work_dir,
        config_path,
        num_speakers,
        args.device,
        args.batch_size,
        args.num_workers,
        external_vad_manifest,
    )

    cfg = OmegaConf.load(config_path)
    diarizer = diarizer_class(cfg=cfg)
    with torch_module.inference_mode():
        diarizer.diarize()

    rttm_files = sorted(work_dir.rglob("*.rttm"))
    print("RTTM files:", [str(path) for path in rttm_files], flush=True)
    if rttm_files:
        turns = parse_rttm(rttm_files[0])
        json_path = work_dir / "speaker_turns.json"
        json_path.write_text(json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {"turn_count": len(turns), "json": str(json_path), "rttm": str(rttm_files[0])},
                ensure_ascii=False,
            ),
            flush=True,
        )


def serve(torch_module, diarizer_class) -> None:
    print(json.dumps({"type": "ready"}), flush=True)
    for line in iter(input, ""):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            run_job(SimpleNamespace(**payload), torch_module, diarizer_class)
            print(json.dumps({"type": "job_complete"}), flush=True)
        except Exception as exc:
            traceback.print_exc()
            print(json.dumps({"type": "job_error", "error": str(exc)}, ensure_ascii=False), flush=True)
        finally:
            gc.collect()
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--speakers", default="auto")
    parser.add_argument("--work-dir", default="/workspace/nemo_out")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--vad-segments-json")
    args = parser.parse_args()

    import torch
    from nemo.collections.asr.models import ClusteringDiarizer

    install_fast_vad_smoothing(torch)
    if args.server:
        serve(torch, ClusteringDiarizer)
    else:
        validate_args(args, parser)
        run_job(args, torch, ClusteringDiarizer)


if __name__ == "__main__":
    main()
