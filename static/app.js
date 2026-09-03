const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const form = $("#uploadForm");
const fileInput = $("#fileInput");
const dropzone = $("#dropzone");
const fileName = $("#fileName");
const languageInput = $("#languageInput");
const diarizeInput = $("#diarizeInput");
const diarizationEngineSelect = $("#diarizationEngineSelect");
const speakersInput = $("#speakersInput");
const engineWrap = $("#engineWrap");
const speakersWrap = $("#speakersWrap");
const submitBtn = $("#submitBtn");
const statusText = $("#statusText");
const percentText = $("#percentText");
const progressBar = $("#progressBar");
const result = $("#result");
const preview = $("#preview");
const resultMeta = $("#resultMeta");
const copyBtn = $("#copyBtn");
const formatSelect = $("#formatSelect");
const viewSelect = $("#viewSelect");
const downloadBtn = $("#downloadBtn");
const transcribeProgress = $("#transcribeProgress");

const convertForm = $("#convertForm");
const convertFileInput = $("#convertFileInput");
const convertDropzone = $("#convertDropzone");
const convertFileName = $("#convertFileName");
const audioFormatSelect = $("#audioFormatSelect");
const convertSubmitBtn = $("#convertSubmitBtn");
const convertStatusText = $("#convertStatusText");
const convertPercentText = $("#convertPercentText");
const convertProgressBar = $("#convertProgressBar");
const convertProgress = $("#convertProgress");
const convertResult = $("#convertResult");
const convertDownloadBtn = $("#convertDownloadBtn");

const runtimeBadge = $("#runtimeBadge");
const settingsStatus = $("#settingsStatus");
const modelSelect = $("#modelSelect");
const deviceSelect = $("#deviceSelect");
const computeTypeSelect = $("#computeTypeSelect");
const beamSizeInput = $("#beamSizeInput");
const batchedInput = $("#batchedInput");
const batchSizeInput = $("#batchSizeInput");
const cpuThreadsInput = $("#cpuThreadsInput");
const numWorkersInput = $("#numWorkersInput");
const vadSilenceInput = $("#vadSilenceInput");
const vramInput = $("#vramInput");
const profileSelect = $("#profileSelect");
const defaultDiarizationSelect = $("#defaultDiarizationSelect");
const defaultSpeakersInput = $("#defaultSpeakersInput");
const nemoBatchSizeInput = $("#nemoBatchSizeInput");
const nemoNumWorkersInput = $("#nemoNumWorkersInput");
const nemoReuseVadInput = $("#nemoReuseVadInput");
const maxUploadInput = $("#maxUploadInput");
const engineInputs = $$("input[name='transcriptionEngine']");
const whisperRuntimeDetails = $("#whisperRuntimeDetails");
const gigaamRuntimeDetails = $("#gigaamRuntimeDetails");
const gigaamModelSelect = $("#gigaamModelSelect");
const gigaamDeviceSelect = $("#gigaamDeviceSelect");
const gigaamBatchSizeInput = $("#gigaamBatchSizeInput");
const gigaamVadThresholdInput = $("#gigaamVadThresholdInput");
const gigaamVadSilenceInput = $("#gigaamVadSilenceInput");
const gigaamMaxSegmentInput = $("#gigaamMaxSegmentInput");

const text = {
  noStatus: "Не удалось получить статус",
  fileFallback: "MP3, WAV, M4A, OGG, WEBM, MP4, MOV, MKV и AVI",
  videoFallback: "MP4, MOV, MKV, WEBM или AVI",
  chooseFile: "Выберите файл",
  chooseVideo: "Выберите видео",
  uploadFile: "Загрузка файла",
  uploadVideo: "Загрузка видео",
  whisperStart: "Файл принят, ожидаю GPU",
  ffmpegStart: "Файл принят, запускаю ffmpeg",
  error: "Ошибка",
};

let currentJobId = null;
let pollTimer = null;
let currentConvertId = null;
let convertPollTimer = null;
let settingsData = null;
let copyResetTimer = null;

function activateTab(name) {
  $$(".tab").forEach((tab) => {
    const active = tab.dataset.tab === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  $$(".tab-panel").forEach((panel) => {
    const active = panel.dataset.panel === name;
    panel.hidden = !active;
    panel.classList.toggle("active", active);
  });
  history.replaceState(null, "", `#${name}`);
  if (name === "settings") loadSettings();
}

$$(".tab").forEach((tab) => tab.addEventListener("click", () => activateTab(tab.dataset.tab)));

function setProgress(bar, percentEl, statusEl, wrap, percent, message, state) {
  const normalized = Math.max(0, Math.min(100, Number(percent) || 0));
  bar.style.width = `${normalized}%`;
  bar.parentElement.setAttribute("aria-valuenow", String(normalized));
  percentEl.textContent = `${normalized}%`;
  statusEl.textContent = message || "";
  wrap.classList.toggle("done", state === "done");
  wrap.classList.toggle("error", state === "error");
}

function formatFileLabel(file) {
  return `${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`;
}

function setInputFile(input, file) {
  const transfer = new DataTransfer();
  transfer.items.add(file);
  input.files = transfer.files;
  input.dispatchEvent(new Event("change", { bubbles: true }));
}

function attachDropzone(zone, input) {
  ["dragenter", "dragover"].forEach((eventName) => {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.remove("dragover");
    });
  });
  zone.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (file) setInputFile(input, file);
  });
}

async function readError(response) {
  try {
    const body = await response.json();
    return body.detail || JSON.stringify(body);
  } catch {
    return response.statusText || `HTTP ${response.status}`;
  }
}

function stopPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
}

function stopConvertPolling() {
  if (convertPollTimer) clearInterval(convertPollTimer);
  convertPollTimer = null;
}

function setTranscriptionRunning(running) {
  submitBtn.disabled = running;
  diarizeInput.disabled = running;
  diarizationEngineSelect.disabled = running;
  speakersInput.disabled = running;
}

function diarizationSummary(job) {
  if (!job.diarize) return "Диаризация выключена";
  const engine = job.diarization_engine === "pyannote" ? "pyannote.audio" : "NVIDIA NeMo";
  const requested = job.speakers ? `задано: ${job.speakers}` : "число спикеров: авто";
  const detected = job.speakers_detected ? `, найдено: ${job.speakers_detected}` : "";
  return `${engine} · ${requested}${detected}`;
}

async function copyPreview() {
  if (!preview.value) return;
  let copied = false;
  try {
    await navigator.clipboard.writeText(preview.value);
    copied = true;
  } catch {
    const selectionStart = preview.selectionStart;
    const selectionEnd = preview.selectionEnd;
    preview.focus();
    preview.select();
    copied = document.execCommand("copy");
    preview.setSelectionRange(selectionStart, selectionEnd);
  }
  copyBtn.textContent = copied ? "Скопировано" : "Не удалось скопировать";
  clearTimeout(copyResetTimer);
  copyResetTimer = setTimeout(() => { copyBtn.textContent = "Копировать текст"; }, 1800);
}

async function pollJob() {
  if (!currentJobId) return;
  const response = await fetch(`/api/jobs/${currentJobId}`);
  if (!response.ok) {
    setProgress(progressBar, percentText, statusText, transcribeProgress, 0, text.noStatus, "error");
    stopPolling();
    setTranscriptionRunning(false);
    return;
  }
  const job = await response.json();
  setProgress(progressBar, percentText, statusText, transcribeProgress, job.percent, job.message, job.status);
  transcribeProgress.classList.toggle("model-download", job.phase === "model_download");
  if (job.status === "done") {
    stopPolling();
    setTranscriptionRunning(false);
    result.hidden = false;
    preview.value = job.text_preview || "";
    resultMeta.textContent = diarizationSummary(job);
  } else if (job.status === "error") {
    stopPolling();
    setTranscriptionRunning(false);
    preview.value = job.error || "";
    resultMeta.textContent = "Транскрипция завершилась с ошибкой";
    result.hidden = false;
  }
}

async function pollConversion() {
  if (!currentConvertId) return;
  const response = await fetch(`/api/convert/${currentConvertId}`);
  if (!response.ok) {
    setProgress(convertProgressBar, convertPercentText, convertStatusText, convertProgress, 0, text.noStatus, "error");
    stopConvertPolling();
    convertSubmitBtn.disabled = false;
    return;
  }
  const job = await response.json();
  setProgress(convertProgressBar, convertPercentText, convertStatusText, convertProgress, job.percent, job.message, job.status);
  if (job.status === "done" || job.status === "error") {
    stopConvertPolling();
    convertSubmitBtn.disabled = false;
    convertResult.hidden = false;
  }
}

function fillRuntime(runtime) {
  modelSelect.value = runtime.model;
  deviceSelect.value = runtime.device;
  computeTypeSelect.value = runtime.compute_type;
  beamSizeInput.value = runtime.beam_size;
  batchedInput.checked = Boolean(runtime.batched_inference);
  batchSizeInput.value = runtime.batch_size;
  cpuThreadsInput.value = runtime.cpu_threads;
  numWorkersInput.value = runtime.num_workers;
  vadSilenceInput.value = runtime.vad_min_silence_ms ?? 500;
  batchSizeInput.disabled = !batchedInput.checked;
}

function selectedEngine() {
  return engineInputs.find((input) => input.checked)?.value || "whisper";
}

function syncEngineUI() {
  const engine = selectedEngine();
  $$('[data-engine-option]').forEach((option) => {
    option.classList.toggle("selected", option.dataset.engineOption === engine);
  });
  whisperRuntimeDetails.hidden = engine !== "whisper";
  gigaamRuntimeDetails.hidden = engine !== "gigaam";
  languageInput.disabled = engine === "gigaam";
  languageInput.placeholder = engine === "gigaam" ? "ru (фиксировано)" : "auto или ru";
}

async function loadSettings() {
  try {
    const response = await fetch("/api/settings");
    if (!response.ok) throw new Error(await readError(response));
    settingsData = await response.json();
    const { config, hardware, runtime, storage } = settingsData;
    const tc = config.transcriptor;
    const dc = config.diarization;
    const gigaam = config.gigaam || {};

    modelSelect.replaceChildren(...settingsData.model_catalog.map((model) => {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = `${model.label} · VRAM от ${model.recommended_vram_gb} ГБ`;
      return option;
    }));
    fillRuntime(tc);
    const engineInput = engineInputs.find((input) => input.value === (tc.engine || "whisper"));
    if (engineInput) engineInput.checked = true;
    gigaamModelSelect.replaceChildren(...settingsData.gigaam_model_catalog.map((model) => {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = `${model.label} · VRAM от ${model.recommended_vram_gb} ГБ`;
      return option;
    }));
    gigaamModelSelect.value = gigaam.model ?? "v3_e2e_rnnt";
    gigaamDeviceSelect.value = gigaam.device ?? "cuda";
    gigaamBatchSizeInput.value = gigaam.batch_size ?? 4;
    gigaamVadThresholdInput.value = gigaam.vad_threshold ?? 0.5;
    gigaamVadSilenceInput.value = gigaam.vad_min_silence_ms ?? 500;
    gigaamMaxSegmentInput.value = gigaam.max_segment_seconds ?? 23;
    syncEngineUI();
    vramInput.value = config.hardware?.vram_gb ?? hardware.vram_gb;
    profileSelect.value = config.hardware?.profile ?? "balanced";
    defaultDiarizationSelect.value = dc.default_engine ?? "nemo";
    defaultSpeakersInput.value = dc.default_speakers ?? 0;
    nemoBatchSizeInput.value = dc.batch_size ?? 128;
    nemoNumWorkersInput.value = dc.num_workers ?? 0;
    nemoReuseVadInput.checked = dc.reuse_transcription_vad !== false;
    maxUploadInput.value = config.max_upload_mb ?? 2048;
    diarizationEngineSelect.value = dc.default_engine ?? "nemo";
    speakersInput.value = dc.default_speakers ?? 0;
    diarizeInput.checked = Boolean(dc.enabled_by_default);
    engineWrap.hidden = !diarizeInput.checked;
    speakersWrap.hidden = !diarizeInput.checked;

    $("#gpuName").textContent = hardware.gpu_name || "GPU не обнаружен";
    $("#detectedVram").textContent = `${hardware.vram_gb.toFixed(1)} ГБ`;
    $("#cpuCores").textContent = `${hardware.cpu_logical_cores} потоков`;
    $("#storageUsage").textContent = `${storage.total_gb.toFixed(2)} ГБ · ${storage.file_count} файлов`;
    $("#whisperRuntimeHint").textContent = runtime.gpu_busy
      ? `GPU занят, в очереди: ${runtime.queued_jobs}`
      : `GPU свободен, в очереди: ${runtime.queued_jobs}`;
    $("#gigaamRuntimeHint").textContent = runtime.gpu_busy
      ? `GPU занят, в очереди: ${runtime.queued_jobs}`
      : `GPU свободен, batch ${gigaam.batch_size}`;
    const gigaamAvailability = $("#gigaamAvailability");
    gigaamAvailability.hidden = settingsData.gigaam_available;
    gigaamAvailability.textContent = settingsData.gigaam_available
      ? ""
      : "GigaAM runtime не найден. Выберите Whisper или восстановите локальное окружение GigaAM.";
    const whisperModelLabel = settingsData.model_catalog
      .find((model) => model.id === tc.model)?.label.replace(/^Whisper\s+/, "") ?? tc.model;
    const gigaamModelLabel = settingsData.gigaam_model_catalog
      .find((model) => model.id === gigaam.model)?.label.replace(/^GigaAM\s+/, "") ?? gigaam.model;
    runtimeBadge.textContent = tc.engine === "gigaam"
      ? `GigaAM · ${gigaamModelLabel} · batch ${gigaam.batch_size} · ${runtime.gpu_busy ? "GPU занят" : "GPU свободен"}`
      : `Whisper · ${whisperModelLabel} · batch ${tc.batched_inference ? tc.batch_size : "off"} · ${runtime.gpu_busy ? "GPU занят" : "GPU свободен"}`;
    settingsStatus.textContent = "";
  } catch (error) {
    runtimeBadge.textContent = "Конфигурация недоступна";
    settingsStatus.textContent = `${text.error}: ${error.message}`;
    settingsStatus.className = "status-error";
  }
}

async function applyProfile() {
  const profile = profileSelect.value;
  if (profile === "manual") return;
  settingsStatus.textContent = "Рассчитываю профиль...";
  try {
    const query = new URLSearchParams({ vram_gb: vramInput.value, profile });
    const response = await fetch(`/api/settings/recommend?${query}`);
    if (!response.ok) throw new Error(await readError(response));
    const recommended = await response.json();
    recommended.vad_min_silence_ms = Number(vadSilenceInput.value || 500);
    fillRuntime(recommended);
    settingsStatus.textContent = "Профиль применён к форме. Сохраните изменения.";
    settingsStatus.className = "";
  } catch (error) {
    settingsStatus.textContent = `${text.error}: ${error.message}`;
    settingsStatus.className = "status-error";
  }
}

async function saveSettings() {
  settingsStatus.textContent = "Сохраняю конфигурацию...";
  $("#saveSettingsBtn").disabled = true;
  const payload = {
    hardware: {
      profile: profileSelect.value,
      vram_gb: Number(vramInput.value),
    },
    transcriptor: {
      engine: selectedEngine(),
      model: modelSelect.value,
      device: deviceSelect.value,
      compute_type: computeTypeSelect.value,
      beam_size: Number(beamSizeInput.value),
      batched_inference: batchedInput.checked,
      batch_size: Number(batchSizeInput.value),
      cpu_threads: Number(cpuThreadsInput.value),
      num_workers: Number(numWorkersInput.value),
      device_index: 0,
      vad_filter: true,
      vad_min_silence_ms: Number(vadSilenceInput.value),
    },
    gigaam: {
      model: gigaamModelSelect.value,
      device: gigaamDeviceSelect.value,
      batch_size: Number(gigaamBatchSizeInput.value),
      vad_threshold: Number(gigaamVadThresholdInput.value),
      vad_min_silence_ms: Number(gigaamVadSilenceInput.value),
      max_segment_seconds: Number(gigaamMaxSegmentInput.value),
    },
    diarization: {
      default_engine: defaultDiarizationSelect.value,
      default_speakers: Number(defaultSpeakersInput.value),
      device: "cuda",
      batch_size: Number(nemoBatchSizeInput.value),
      num_workers: Number(nemoNumWorkersInput.value),
      reuse_transcription_vad: nemoReuseVadInput.checked,
    },
    max_upload_mb: Number(maxUploadInput.value),
  };
  try {
    const response = await fetch("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(await readError(response));
    settingsData = await response.json();
    await loadSettings();
    settingsStatus.textContent = "Сохранено. Новая модель загрузится при следующем задании.";
    settingsStatus.className = "status-ok";
  } catch (error) {
    settingsStatus.textContent = `${text.error}: ${error.message}`;
    settingsStatus.className = "status-error";
  } finally {
    $("#saveSettingsBtn").disabled = false;
  }
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  fileName.textContent = file ? formatFileLabel(file) : text.fileFallback;
});
convertFileInput.addEventListener("change", () => {
  const file = convertFileInput.files[0];
  convertFileName.textContent = file ? formatFileLabel(file) : text.videoFallback;
});
attachDropzone(dropzone, fileInput);
attachDropzone(convertDropzone, convertFileInput);

diarizeInput.addEventListener("change", () => {
  engineWrap.hidden = !diarizeInput.checked;
  speakersWrap.hidden = !diarizeInput.checked;
});

copyBtn.addEventListener("click", copyPreview);
batchedInput.addEventListener("change", () => {
  batchSizeInput.disabled = !batchedInput.checked;
  profileSelect.value = "manual";
});
engineInputs.forEach((input) => input.addEventListener("change", syncEngineUI));
deviceSelect.addEventListener("change", () => {
  if (deviceSelect.value === "cpu") {
    computeTypeSelect.value = "int8";
    batchedInput.checked = false;
    batchSizeInput.disabled = true;
  }
});
[$("#modelSelect"), deviceSelect, computeTypeSelect, beamSizeInput, batchSizeInput, cpuThreadsInput, numWorkersInput]
  .forEach((input) => input.addEventListener("change", () => { profileSelect.value = "manual"; }));

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    setProgress(progressBar, percentText, statusText, transcribeProgress, 0, text.chooseFile, "error");
    return;
  }
  stopPolling();
  result.hidden = true;
  preview.value = "";
  resultMeta.textContent = "";
  setTranscriptionRunning(true);
  setProgress(progressBar, percentText, statusText, transcribeProgress, 1, text.uploadFile, "running");
  const data = new FormData();
  data.append("file", file);
  if (languageInput.value.trim()) data.append("language", languageInput.value.trim());
  data.append("diarize", diarizeInput.checked ? "true" : "false");
  data.append("diarization_engine", diarizationEngineSelect.value);
  data.append("speakers", speakersInput.value || "0");
  try {
    const response = await fetch("/api/jobs", { method: "POST", body: data });
    if (!response.ok) throw new Error(await readError(response));
    currentJobId = (await response.json()).job_id;
    setProgress(progressBar, percentText, statusText, transcribeProgress, 3, text.whisperStart, "running");
    pollTimer = setInterval(pollJob, 1000);
    await pollJob();
  } catch (error) {
    setProgress(progressBar, percentText, statusText, transcribeProgress, 0, `${text.error}: ${error.message}`, "error");
    setTranscriptionRunning(false);
  }
});

convertForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = convertFileInput.files[0];
  if (!file) {
    setProgress(convertProgressBar, convertPercentText, convertStatusText, convertProgress, 0, text.chooseVideo, "error");
    return;
  }
  stopConvertPolling();
  convertResult.hidden = true;
  convertSubmitBtn.disabled = true;
  setProgress(convertProgressBar, convertPercentText, convertStatusText, convertProgress, 1, text.uploadVideo, "running");
  const data = new FormData();
  data.append("file", file);
  data.append("audio_format", audioFormatSelect.value);
  try {
    const response = await fetch("/api/convert", { method: "POST", body: data });
    if (!response.ok) throw new Error(await readError(response));
    currentConvertId = (await response.json()).job_id;
    setProgress(convertProgressBar, convertPercentText, convertStatusText, convertProgress, 3, text.ffmpegStart, "running");
    convertPollTimer = setInterval(pollConversion, 1000);
    await pollConversion();
  } catch (error) {
    setProgress(convertProgressBar, convertPercentText, convertStatusText, convertProgress, 0, `${text.error}: ${error.message}`, "error");
    convertSubmitBtn.disabled = false;
  }
});

downloadBtn.addEventListener("click", () => {
  if (!currentJobId) return;
  window.location.href = `/api/jobs/${currentJobId}/download?format=${encodeURIComponent(formatSelect.value)}&view=${encodeURIComponent(viewSelect.value)}`;
});
convertDownloadBtn.addEventListener("click", () => {
  if (currentConvertId) window.location.href = `/api/convert/${currentConvertId}/download`;
});
$("#applyProfileBtn").addEventListener("click", applyProfile);
$("#saveSettingsBtn").addEventListener("click", saveSettings);
$("#refreshSettingsBtn").addEventListener("click", loadSettings);

const initialTab = location.hash.slice(1);
activateTab(["transcription", "converter", "settings"].includes(initialTab) ? initialTab : "transcription");
loadSettings();
