# Local Speech Studio

[Русский](#русский) | [English](#english)

## Русский

Local Speech Studio - локальное Windows-приложение для распознавания речи, разделения спикеров и конвертации видео в аудио. Файлы остаются на вашем компьютере. Веб-интерфейс поддерживает drag-and-drop, видимый прогресс заданий и экспорт в TXT, Markdown или PDF с корректной кириллицей.

### Возможности

- Переключение между движками Whisper и GigaAM. Конкретная модель выбирается в параметрах runtime выбранного движка.
- GigaAM рекомендуется для русских встреч; Whisper быстрее и поддерживает больше языков.
- Модель выбранного движка автоматически скачивается при первой транскрипции.
- В прогрессе отдельно показываются скачивание и загрузка модели с точным названием.
- Пакетный GPU-инференс и аппаратные профили по доступной VRAM.
- Диаризация спикеров через NVIDIA NeMo или pyannote.
- Аудио и видео: MP3, WAV, M4A, OGG, WEBM, MP4, MOV, MKV и AVI.
- Конвертер видео в MP3, WAV или M4A.
- Простой текст или полный ответ движка в TXT, MD и PDF.
- PDF с Unicode-шрифтом, группировкой абзацев и переносом длинного JSON.

### Требования

- Windows 10/11
- 64-битный Python 3.10-3.12
- ffmpeg в `PATH`
- Рекомендуется NVIDIA GPU с актуальным драйвером
- Для Whisper Turbo рекомендуется от 8 ГБ VRAM
- CPU-режим поддерживается, но работает существенно медленнее

При необходимости установите ffmpeg:

```powershell
winget install Gyan.FFmpeg
```

### Быстрая установка

```powershell
git clone https://github.com/Vkabuto23/local-speech-studio.git
cd local-speech-studio
.\install.bat
```

После установки запустите сервис:

```powershell
.\start.bat
```

Откройте [http://127.0.0.1:8015](http://127.0.0.1:8015).

Установщик создаёт `.venv`, ставит протестированную сборку PyTorch с CUDA 12.8, оба движка распознавания и NeMo, затем создаёт локальный `config.json`. Docker не нужен.

Установка только для CPU:

```powershell
.\setup.ps1 -CpuOnly -SkipNeMo
```

### Первый запуск и скачивание моделей

Модели намеренно не хранятся в Git-репозитории. Выберите движок во вкладке **Настройки**, загрузите медиафайл и нажмите **Запустить транскрипцию**. Если настроенной модели нет в кэше, задание перейдёт в заметную фазу `model_download` и покажет точное название модели. После скачивания распознавание начнётся автоматически.

Стандартные каталоги кэша:

- Whisper: `%USERPROFILE%\.cache\huggingface\hub`
- GigaAM: `%USERPROFILE%\.cache\gigaam`
- NeMo и pyannote: пользовательские кэши соответствующих библиотек

Первый запуск может занять несколько минут. Следующие задания используют локальный кэш.

### Выбор движка

Рекомендуемый запас VRAM для GPU-режима со стандартными настройками:

| Модель | Рекомендуемая VRAM |
| --- | ---: |
| Whisper large-v3 | от 12 ГБ |
| Whisper large-v3-turbo | от 8 ГБ |
| Whisper medium | от 5 ГБ |
| Whisper small | от 3 ГБ |
| Whisper base | от 2 ГБ |
| GigaAM V3 E2E RNNT | от 4 ГБ |
| GigaAM V3 E2E CTC | от 4 ГБ |

При увеличении batch size может потребоваться больше памяти. В CPU-режиме VRAM не используется.

#### GigaAM

Движок для русской речи. В runtime доступны модели GigaAM V3 E2E RNNT и GigaAM V3 E2E CTC. Они выдают пунктуацию, нормализованный текст и таймкоды слов. Длинная запись режется через Silero VAD и обрабатывается батчами в отдельном процессе. После задания процесс завершается и освобождает RAM и VRAM.

#### Whisper

Многоязычный движок на faster-whisper и CTranslate2. В runtime доступны модели large-v3, large-v3-turbo, medium, small и base, а также batch size, тип вычислений, beam size, CPU threads и model workers.

### Разделение спикеров

Включите **Разбить по спикерам**, выберите NeMo или pyannote и выберите автоматическое определение либо точное ожидаемое число участников. Если число участников известно, точное значение обычно надёжнее автоматической кластеризации. Оба движка распознавания отдают таймкоды слов, которые использует общий этап назначения спикеров.

Предпросмотр на экране содержит полную транскрипцию. Весь текст можно сразу скопировать без скачивания файла.

NeMo устанавливается по умолчанию на системах с NVIDIA GPU. Для некоторых моделей pyannote нужен read-токен Hugging Face и принятие условий модели. Скопируйте `.env.example` в `.env` и заполните:

```dotenv
HF_TOKEN=hf_ваш_read_токен
```

Файл `.env` исключён из Git. Никогда не публикуйте его.

### Экспорт

- `simple`: только читаемый текст, включая спикеров при включённой диаризации.
- `full`: метаданные, сегменты с таймкодами, слова и полный ответ движка.
- Форматы: TXT и Markdown с UTF-8 BOM, а также A4 PDF со встроенным Unicode-шрифтом.

### Конфигурация

Публичные значения находятся в `config.example.json`. При первом запуске они копируются в локальный `config.json`, который исключён из Git. Обычные изменения выполняются через вкладку **Настройки**.

Аппаратные профили автоматически настраивают Whisper по доступной VRAM. У GigaAM отдельные параметры модели, устройства, batch size, VAD threshold, длительности тишины и максимального фрагмента.

### Приватность

- Сервер по умолчанию доступен только через `127.0.0.1`.
- Загруженные медиа и результаты исключены из Git.
- После скачивания моделей распознавание выполняется локально.
- В приложении нет телеметрии.
- К провайдерам моделей уходят только обычные запросы скачивания отсутствующей модели.

### Решение проблем

- `ffmpeg is not available`: установите ffmpeg и перезапустите терминал.
- Ошибка CUDA: обновите драйвер NVIDIA или выберите CPU в настройках.
- Не найден GigaAM: выполните `.\setup_gigaam.ps1 -TargetPython .\.venv\Scripts\python.exe`.
- Не найден NeMo: выполните `.\.venv\Scripts\python.exe -m pip install "nemo_toolkit[asr]==2.7.3"`.
- Ошибка gated-модели pyannote: примите условия на Hugging Face и добавьте `HF_TOKEN` в `.env`.

### Лицензия и модели

Код приложения распространяется по лицензии MIT. Скачиваемые модели не входят в репозиторий и используют собственные лицензии и условия. GigaAM разработан [Salute Developers](https://github.com/salute-developers/GigaAM), Whisper разработан OpenAI, а локальные CTranslate2-конверсии используются через faster-whisper.

Полная русская версия также доступна отдельным файлом: [README.ru.md](README.ru.md).

---

## English

Local Speech Studio is a self-hosted Windows application for local speech-to-text, speaker diarization, and video-to-audio conversion. Files stay on your computer. The web interface supports drag and drop, visible job progress, and TXT, Markdown, or Unicode PDF exports.

## Highlights

- Switch between the Whisper and GigaAM engines. Select a concrete model in the runtime settings for the active engine.
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

Recommended VRAM headroom for GPU mode with the default settings:

| Model | Recommended VRAM |
| --- | ---: |
| Whisper large-v3 | 12 GB or more |
| Whisper large-v3-turbo | 8 GB or more |
| Whisper medium | 5 GB or more |
| Whisper small | 3 GB or more |
| Whisper base | 2 GB or more |
| GigaAM V3 E2E RNNT | 4 GB or more |
| GigaAM V3 E2E CTC | 4 GB or more |

Larger batch sizes can require additional memory. CPU mode does not use VRAM.

### GigaAM

An engine for Russian speech. Its runtime provides the GigaAM V3 E2E RNNT and GigaAM V3 E2E CTC models. They provide punctuation, text normalization, and word timestamps. Long audio is split with Silero VAD and processed in batches by a per-job subprocess, releasing host and GPU memory after completion.

### Whisper

A multilingual engine based on faster-whisper and CTranslate2. Its runtime provides large-v3, large-v3-turbo, medium, small, and base, plus configurable batch size, compute type, beam size, and worker count.

## Speaker Diarization

Enable **Split by speakers**, select NeMo or pyannote, and choose automatic speaker counting or an exact expected number. When the participant count is known, entering the exact number is usually more reliable than automatic clustering. Both transcription engines produce word timestamps used by the common speaker-assignment stage.

The on-screen preview contains the complete transcript and can be copied directly without downloading a file.

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

## Troubleshooting

- `ffmpeg is not available`: install ffmpeg and restart the terminal.
- CUDA error: update the NVIDIA driver or choose CPU in Settings.
- GigaAM runtime missing: run `.\setup_gigaam.ps1 -TargetPython .\.venv\Scripts\python.exe`.
- NeMo missing: run `.\.venv\Scripts\python.exe -m pip install "nemo_toolkit[asr]==2.7.3"`.
- A gated pyannote model fails: accept its Hugging Face terms and set `HF_TOKEN` in `.env`.

## License and Models

The application code is licensed under the MIT License. Downloaded models are not part of this repository and remain subject to their respective licenses and terms. GigaAM is developed by [Salute Developers](https://github.com/salute-developers/GigaAM); Whisper is developed by OpenAI and the local CTranslate2 conversions are provided through the faster-whisper ecosystem.
