import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from faster_whisper import BatchedInferencePipeline, WhisperModel


def read_gpu() -> tuple[int, int]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        memory, utilization = output.splitlines()[0].split(",")
        return int(memory.strip()), int(utilization.strip())
    except Exception:
        return 0, 0


def monitor_gpu(stop: threading.Event, samples: list[tuple[int, int]]) -> None:
    while not stop.wait(0.2):
        samples.append(read_gpu())


def benchmark(args: argparse.Namespace) -> list[dict]:
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        cpu_threads=args.cpu_threads,
        num_workers=args.num_workers,
    )
    pipeline = BatchedInferencePipeline(model=model)
    results = []

    for batch_size in args.batch_sizes:
        samples: list[tuple[int, int]] = []
        stop = threading.Event()
        monitor = threading.Thread(target=monitor_gpu, args=(stop, samples), daemon=True)
        monitor.start()
        started = time.perf_counter()
        segments, info = pipeline.transcribe(
            str(args.audio),
            language=args.language,
            beam_size=args.beam_size,
            batch_size=batch_size,
            vad_filter=True,
            word_timestamps=True,
        )
        segment_count = sum(1 for _ in segments)
        elapsed = time.perf_counter() - started
        stop.set()
        monitor.join(timeout=1)
        duration = float(info.duration)
        results.append(
            {
                "batch_size": batch_size,
                "audio_seconds": round(duration, 2),
                "elapsed_seconds": round(elapsed, 2),
                "realtime_factor": round(elapsed / duration, 4),
                "speed_x": round(duration / elapsed, 2),
                "segments": segment_count,
                "peak_gpu_memory_mib": max((sample[0] for sample in samples), default=0),
                "peak_gpu_utilization_percent": max((sample[1] for sample in samples), default=0),
            }
        )
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark faster-whisper batch sizes on local hardware.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--language", default="ru")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--cpu-threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.audio.exists():
        parser.error(f"Audio file does not exist: {args.audio}")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    benchmark_results = benchmark(parsed)
    if parsed.output:
        parsed.output.parent.mkdir(parents=True, exist_ok=True)
        parsed.output.write_text(json.dumps(benchmark_results, ensure_ascii=False, indent=2), encoding="utf-8")
