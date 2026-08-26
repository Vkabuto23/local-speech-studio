# NeMo diarization worker

This directory contains the internal NVIDIA NeMo worker used by Local Speech Studio.

Install all application dependencies from the repository root:

```powershell
.\setup.ps1
```

The application starts this worker automatically when NeMo diarization is selected. Temporary manifests, WAV files, embeddings, RTTM files, and JSON results are written to `runs/` and removed after successful jobs unless `diarization.keep_artifacts` is enabled in the local configuration.

No Docker runtime is required.
