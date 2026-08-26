import argparse
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import DataLoader

import gigaam
from gigaam.utils import AudioDataset


def emit(percent: int, message: str) -> None:
    print(json.dumps({"type": "progress", "percent": percent, "message": message}, ensure_ascii=False), flush=True)


def model_cached(model_name: str) -> bool:
    cache_dir = Path(os.getenv("GIGAAM_CACHE_DIR", "~/.cache/gigaam")).expanduser()
    checkpoint = cache_dir / f"{model_name}.ckpt"
    tokenizer = cache_dir / f"{model_name}_tokenizer.model"
    return checkpoint.is_file() and ("e2e" not in model_name or tokenizer.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-wav", required=True)
    parser.add_argument("--boundaries-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model", default="v3_e2e_rnnt")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    started = time.time()
    audio, sample_rate = sf.read(args.input_wav, dtype="float32", always_2d=False)
    if sample_rate != 16000:
        raise ValueError(f"Expected 16000 Hz audio, got {sample_rate} Hz")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    boundaries = json.loads(Path(args.boundaries_json).read_text(encoding="utf-8"))
    chunks = [audio[round(item["start"] * sample_rate) : round(item["end"] * sample_rate)] for item in boundaries]
    if not chunks:
        Path(args.output_json).write_text(json.dumps({"segments": [], "word_segments": []}), encoding="utf-8")
        return

    if model_cached(args.model):
        emit(20, f"Загружаю модель GigaAM {args.model}")
    else:
        emit(20, f"Скачиваю модель GigaAM {args.model}. Не закрывайте приложение")
    model = gigaam.load_model(
        args.model,
        fp16_encoder=args.device == "cuda",
        use_flash=False,
        device=args.device,
        download_root=str(Path(os.getenv("GIGAAM_CACHE_DIR", "~/.cache/gigaam")).expanduser()),
    )
    emit(30, f"GigaAM готов: {len(chunks)} речевых фрагментов")

    dataset = AudioDataset(chunks, tokenizer=None)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=AudioDataset.collate,
        num_workers=0,
    )

    segments = []
    words_payload = []
    chunk_index = 0
    total_batches = len(loader)
    with torch.inference_mode():
        for batch_index, (wav_pad, wav_lens) in enumerate(loader, start=1):
            wav_pad = wav_pad.to(model._device).to(model._dtype)
            wav_lens = wav_lens.to(model._device)
            encoded, encoded_len = model.forward(wav_pad, wav_lens)
            decoded = model._decode(encoded, encoded_len, wav_lens, word_timestamps=True)
            for text, words in decoded:
                boundary = boundaries[chunk_index]
                offset = float(boundary["start"])
                clean_text = (text or "").strip()
                segment_words = []
                for word in words or []:
                    item = {
                        "start": round(offset + float(word.start), 3),
                        "end": round(offset + float(word.end), 3),
                        "text": (word.text or "").strip(),
                        "probability": None,
                    }
                    segment_words.append(item)
                    words_payload.append(item)
                segments.append(
                    {
                        "id": chunk_index,
                        "start": round(offset, 3),
                        "end": round(float(boundary["end"]), 3),
                        "text": clean_text,
                        "words": segment_words,
                    }
                )
                chunk_index += 1
            percent = 30 + round(batch_index / total_batches * 65)
            emit(percent, f"GigaAM: батч {batch_index} из {total_batches}")

    duration = len(audio) / sample_rate
    result = {
        "engine": "gigaam",
        "segments": segments,
        "word_segments": words_payload,
        "full_text": " ".join(item["text"] for item in segments if item["text"]).strip(),
        "language": "ru",
        "language_probability": None,
        "duration": round(duration, 2),
        "processing_sec": round(time.time() - started, 2),
        "model": args.model,
        "device": args.device,
        "compute_type": "float16" if args.device == "cuda" else "float32",
        "batched_inference": True,
        "batch_size": args.batch_size,
    }
    Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    emit(100, f"GigaAM завершил распознавание за {result['processing_sec']:.1f} с")


if __name__ == "__main__":
    main()
