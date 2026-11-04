# Architecture

ECG Arrhythmia Studio is split into an analysis pipeline, an API layer and a
dashboard.

## Signal pipeline (`packages/ml-engine`)

1. `preprocessing.py` loads a raw ECG recording, normalises the amplitude and
   slices it into fixed-length windows.
2. `stft_converter.py` converts each window into a time-frequency representation
   used as model input.
3. `model.py` runs the classifier over the converted windows and aggregates the
   per-window scores into a recording-level prediction.
4. `gradcam.py` produces activation overlays so the regions driving a prediction
   can be inspected visually.

## API (`packages/backend`)

The FastAPI application exposes signal upload and prediction endpoints. Long
running analysis is handed to `services/job_runner.py`, which tracks job state so
the dashboard can poll for progress instead of holding a request open.

- `app/core/config.py` holds runtime settings.
- `app/core/database.py` provides persistence for signals and prediction jobs.
- `app/routers/signals.py` handles ingestion.
- `app/routers/predictions.py` handles job creation and result retrieval.

## Dashboard (`packages/frontend`)

A Next.js application that talks to the backend through `app/lib/api.ts`.
Components under `app/components/ui` are presentational, while
`app/components/dashboard` renders the uploaded waveform, the spectrogram and the
Grad-CAM overlay for a completed prediction.

## Local orchestration

`docker-compose.yml` starts the backend and frontend containers together.
`scripts/verify-all.sh` exercises the pipeline, the API and the frontend build in
one pass.
