# Local Speech Studio

[Русская документация](README.ru.md)

Local Speech Studio is a self-hosted Windows application for local speech-to-text, speaker diarization, and video-to-audio conversion. Files stay on your computer. The web interface supports drag and drop, visible job progress, and TXT, Markdown, or Unicode PDF exports.

## Highlights

- Switch between Whisper `large-v3-turbo` and GigaAM V3 E2E RNNT.
- GigaAM is recommended for Russian meetings; Whisper is the multilingual and faster option.
- Recognition models download automatically when the selected engine is used for the first time.
- The job status explicitly shows model download and model loading phases.
- Batched GPU inference with configurable VRAM profiles.
- NVIDIA NeMo or pyannote speaker diarization.
- Audio and video input including MP3, WAV, M4A, OGG, WEBM, MP4, MOV, MKV, and AVI.
- Video-to-MP3, WAV, or M4A converter.
- Simple text or complete engine response in TXT, MD, and PDF.
- Cyrillic-safe PDF generation with paragraph grouping and wrapped raw JSON.

## Requirements

- Windows 10/11
- Python 3.10-3.12, 64-bit
- ffmpeg available in `PATH`
- NVIDIA GPU with a current driver is recommended
- 8 GB VRAM recommended for Whisper Turbo; GigaAM can run with less
- CPU mode is supported but substantially slower

Install ffmpeg with winget if needed:

```powershell
winget install Gyan.FFmpeg
```

## Quick Installation

```powershell
git clone https://github.com/Vkabuto23/local-speech-studio.git
cd local-speech-studio
.\install.bat
```

Then start the service:

```powershell
.\start.bat
```

Open [http://127.0.0.1:8015](http://127.0.0.1:8015).

The installer creates `.venv`, installs a tested CUDA 12.8 PyTorch build, installs both transcription engines and NeMo, and creates a local `config.json`. Docker is not required.

CPU-only installation:

```powershell
.\setup.ps1 -CpuOnly -SkipNeMo
```

## First Transcription and Model Downloads

Models are intentionally not stored in this Git repository. Choose an engine in **Settings**, upload a media file, and click **Start transcription**. If the configured model is missing, the job changes to a visible `model_download` phase and displays the exact model name. Transcription starts automatically after the download completes.

Default cache locations:

- Whisper: `%USERPROFILE%\.cache\huggingface\hub`
- GigaAM: `%USERPROFILE%\.cache\gigaam`
- NeMo and pyannote: framework-managed user caches

The first run can take several minutes depending on the model and connection. Later runs use the local cache.

## Engine Selection

### GigaAM V3 E2E RNNT

Recommended for Russian meetings. It provides punctuation, text normalization, and word timestamps. Long audio is split with Silero VAD and processed in batches by a per-job subprocess, releasing host and GPU memory after completion.

### Whisper large-v3-turbo

Recommended for multilingual audio and maximum throughput. It uses faster-whisper and CTranslate2 with a persistent model cache and configurable batch size, compute type, beam size, and worker count.

## Speaker Diarization

Enable **Split by speakers**, select NeMo or pyannote, and enter the expected number of speakers. Both transcription engines produce word timestamps used by the common speaker-assignment stage.

NeMo is installed by default for NVIDIA GPU systems. Some pyannote models require a Hugging Face read token and acceptance of their model terms. Copy `.env.example` to `.env` and set:

```dotenv
HF_TOKEN=hf_your_read_token
```

Never commit `.env`; it is excluded by `.gitignore`.

## Exports

- `simple`: readable transcript text, including speaker labels when diarization is enabled.
- `full`: metadata, timestamped segments, word timestamps, and the complete engine response.
- Formats: UTF-8 BOM TXT, UTF-8 BOM Markdown, and A4 PDF with embedded Unicode fonts.

## Configuration

The public defaults live in `config.example.json`. On first start they are copied to the ignored local file `config.json`. Use the Settings tab for normal changes.

Hardware profiles configure Whisper automatically from available VRAM. GigaAM has separate model, device, batch, VAD threshold, silence, and maximum segment controls.

## Privacy

- The web server binds to `127.0.0.1` by default.
- Uploaded media and generated files are excluded from Git.
- Recognition runs locally after model download.
- No application telemetry is implemented.
- Model providers receive ordinary download requests only when a model is missing.

## Development

```powershell
python -m unittest discover -s tests -v
```

Run without auto-reload to avoid duplicate GPU model allocation:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8015
```

## Troubleshooting

- `ffmpeg is not available`: install ffmpeg and restart the terminal.
- CUDA error: update the NVIDIA driver or choose CPU in Settings.
- GigaAM runtime missing: run `.\setup_gigaam.ps1 -TargetPython .\.venv\Scripts\python.exe`.
- NeMo missing: run `.\.venv\Scripts\python.exe -m pip install "nemo_toolkit[asr]==2.7.3"`.
- A gated pyannote model fails: accept its Hugging Face terms and set `HF_TOKEN` in `.env`.

## License and Models

The application code is licensed under the MIT License. Downloaded models are not part of this repository and remain subject to their respective licenses and terms. GigaAM is developed by [Salute Developers](https://github.com/salute-developers/GigaAM); Whisper is developed by OpenAI and the local CTranslate2 conversions are provided through the faster-whisper ecosystem.
