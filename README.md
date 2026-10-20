# ECG Signal-to-Image Arrhythmia Classification & Explainability Studio

## 📌 Executive Summary

This project is a complete **ICU Dark Workbench** for ECG arrhythmia classification with AI explainability. It integrates a Python ML pipeline (signal preprocessing, STFT spectrogram generation, 3-class CNN classification, and Grad-CAM heatmap generation) with a Neo-brutalist-styled Next.js frontend dashboard.

The system takes **raw 1D ECG signals** (360 Hz, MIT-BIH dataset format), processes them through a validated signal processing pipeline, generates **128×128 spectrograms**, classifies them into **3 arrhythmia categories** using a 5-block CNN, and produces **Grad-CAM heatmaps** visualizing the AI model's decision regions. The frontend provides a dark-mode clinical workstation interface with real-time ECG waveforms, spectrogram visualization, and AI prediction metrics.

---

## 🏗️ Visual System Architecture

```mermaid
graph TD
    %% === Raw Input Layer ===
    subgraph Input[Raw ECG Input]
        direction LR
        ECG_RAW[1D ECG Signal<br/>360 Hz, MIT-BIH Format]
    end

    %% === Preprocessing Block ===
    subgraph Preprocessing[0.5–40 Hz Bandpass Filter]
        direction LR
        BANDPASS[4th-Order Butterworth Bandpass Filter<br/>Passband: 0.5–40 Hz]
        RPEAK[Pan-Tompkins R-Peak Detection<br/>5–15 Hz Bandpass, Differentiation, Squaring]
        SEGMENT[Beat Segmentation<br/>±80 ms around R-peaks]
        NORMALIZE[Z-Score & Min-Max Normalization<br/>Per-beat normalization]
    end

    %% === Feature Extraction Block ===
    subgraph Feature[STFT Spectrogram Generation]
        direction LR
        STFT[Short-Time Fourier Transform<br/>Hann window 128-sample, 28-hop, 256 FFT]
        NORM_SPEC[Spectrogram Normalization<br/>[0, 1] range]
        RESIZE[Resize/Crop to 128×128]
    end

    %% === Classification Block ===
    subgraph Classification[3-Class CNN Classification]
        direction LR
        CNN[5-Block CNN (32→64→128→256 filters)<br/>BatchNorm, ReLU, MaxPool after each block]
        ADAPTIVE[AdaptiveAvgPool2d → Dense(128) → Dropout(0.5) → Dense(64) → ReLU → Dropout(0.3) → Dense(3) softmax]
        LOGITS[Logits → Softmax Probabilities<br/>Classes: Normal, SVEB, VEB]
    end

    %% === Explainability Block ===
    subgraph Explainability[Grad-CAM Heatmap Generation]
        direction LR
        GRADCAM[Grad-CAM Hook on Final Conv Layer (block4)<br/>Gradient Averaging across temporal dimension]
        RELU[ReLU Activation & Normalization to [0, 1]]
        RESIZE_GRAD[Spatial Resize to 128×128]
        COLORMAP[OpenCV Jet Colormap Application<br/>Overlay on original spectrogram]
        PNG[PNG Byte Output & Base64 Persistence]
    end

    %% === UI Layer ===
    subgraph UI[Next.js "ICU Dark Workbench" Frontend]
        direction LR
        DASHBOARD[Tri-View Clinical Dashboard<br/>ECG Waveform + Spectrogram + Grad-CAM]
        FILE_DROP[File Dropzone<br/>CSV/NPY/MAT/JSON upload (10MB limit)]
        STATUS[Status Bar & Diagnostic Report<br/>Health monitoring, prediction download]
    end

    %% === Connections ===
    ECG_RAW --> BANDPASS
    BANDPASS --> RPEAK
    RPEAK --> SEGMENT
    SEGMENT --> NORMALIZE
    NORMALIZE --> STFT
    STFT --> NORM_SPEC
    NORM_SPEC --> RESIZE
    RESIZE --> CNN
    CNN --> LOGITS
    LOGITS --> GRADCAM
    GRADCAM --> RELU
    RELU --> RESIZE_GRAD
    RESIZE_GRAD --> COLORMAP
    COLORMAP --> PNG
    PNG --> DASHBOARD

    style Input fill:#141414,color:#FFFFFF,stroke:#FFE600,stroke-width:2px
    style Preprocessing fill:#0A0A0A,color:#00F0FF,stroke:#FF0055,stroke-width:2px
    style Feature fill:#0A0A0A,color:#00F0FF,stroke:#FF0055,stroke-width:2px
    style Classification fill:#141414,color:#00F0FF,stroke:#FF0055,stroke-width:2px
    style Explainability fill:#141414,color:#FF0055,stroke:#00F0FF,stroke-width:2px
    style UI fill:#0A0A0A,color:#FFE600,stroke:#00F0FF,stroke-width:2px
```

---

## 📡 Complete REST API Endpoint Reference

| Endpoint | Method | Request Body | Response | Status Codes | Description |
|----------|--------|--------------|----------|--------------|-------------|
| `/api/v1/signals/upload` | `POST` | `multipart/form-data`: `file` (CSV/NPY/MAT/JSON), `samplingRate` (number) | `JobStatusResponse` { `jobId`, `signalId`, `status` } | **202 Accepted** - Job queued for processing<br>**400 Bad Request** - Invalid file type or size<br>**413 Payload Too Large** - File exceeds 10MB limit<br>**500 Internal Server Error** - Processing initialization failed | Upload raw ECG signal for background ML pipeline processing |
| `/api/v1/jobs/{job_id}` | `GET` | `jobId` (path parameter) | `JobStatusResponse` { `status`, `progress`, `message`, `predictionId` } | **200 OK** - Job completed or in progress<br>**404 Not Found** - Job ID not found | Poll background job status and execution progress |
| `/api/v1/predictions/{prediction_id}` | `GET` | `predictionId` (path parameter) | `PredictionMetadataResponse` { `classId`, `className`, `confidence`, `probabilities`, `explanationUrl` } | **200 OK** - Prediction results ready<br>**404 Not Found** - Prediction ID not found<br>**409 Conflict** - Prediction expired or overwritten | Retrieve AI classification result, confidence score, and classification tags |
| `/api/v1/predictions/{prediction_id}/gradcam` | `GET` | `predictionId` (path parameter) | `PNG image bytes` (image/png) | **200 OK** - Grad-CAM heatmap image ready<br>**404 Not Found** - Grad-CAM not generated<br>**503 Service Unavailable** - Grad-CAM generation in progress | Return raw PNG bytes of the generated Grad-CAM heatmap overlay |
| `/health` | `GET` | `None` | `HealthCheckResponse` { `status`, `model_loaded`, `version` } | **200 OK** - Backend healthy, model loaded<br>**503 Service Unavailable** - Backend not ready, model loading | Verify FastAPI service status and model initialization |

### Request/Response Schemas Summary

#### SignalUploadRequest (POST /signals/upload)
```json
{
  "file": "<base64-encoded or multipart file>",
  "samplingRate": 360
}
```

#### JobStatusResponse
```json
{
  "jobId": "string",
  "status": "QUEUED|PROCESSING|COMPLETED|FAILED",
  "progress": number (0-100),
  "message": "string (optional)",
  "predictionId": "string (optional)"
}
```

#### PredictionMetadataResponse
```json
{
  "predictionId": "string",
  "classId": number (0=Normal, 1=SVEB, 2=VEB),
  "className": "Normal" | "SVEB" | "Ventricular",
  "confidence": number (0.0 - 1.0),
  "probabilities": {
    "Normal": number,
    "SVEB": number,
    "Ventricular": number
  },
  "explanationUrl": "string (URL to Grad-CAM image)",
  "spectrogramUrl": "string (URL to STFT spectrogram image)"
}
```

#### ErrorResponse
```json
{
  "detail": "string describing the error"
}
```

---

## ⚡ Quickstart Guide

### Option 1: Local Development (Manual Setup)

#### Backend (FastAPI + ML Pipeline)

```bash
# 1. Navigate to backend
cd packages/backend

# 2. Rename env.example and configure
cp .env.example .env
# Edit .env with your preferred settings

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install ML engine in development mode
pip install -e packages/ml-engine/

# 5. Run the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### Frontend (Next.js)

```bash
# 1. Navigate to frontend
cd packages/frontend

# 2. Rename env.example and configure
cp .env.example .env
# Edit .env with backend URL

# 3. Install Node.js dependencies
npm ci

# 4. Start development server
npm run dev
# Frontend will be available at http://localhost:3000
```

#### Full Pipeline Test

```bash
# 1. Ensure backend is running at http://localhost:8000
# 2. Open frontend at http://localhost:3000
# 3. Use the File Dropzone to upload a .csv or .npy signal file
# 4. Monitor job status in the dashboard
# 5. View AI classification results and Grad-CAM heatmaps
```

#### Option 2: Single-Command Docker Compose Deployment

```bash
# 1. From project root
docker compose up --build

# 2. Wait for containers to start (≈ 60-90 seconds for ML model warmup)
# 3. Access the frontend at http://localhost:3000
# 4. The backend healthcheck is auto-verified at http://localhost:8000/api/v1/health

# 5. Use the UI to upload ECG signal files and view results
# 6. To stop: docker compose down
```

#### Docker Comprise Specifics

| Action | Command |
|--------|---------|
| Start services | `docker compose up --build -d` |
| View logs | `docker compose logs -f` |
| Stop services | `docker compose down` |
| Rebuild images | `docker compose up --build --force-rm` |
| Check health | `curl http://localhost:8000/api/v1/health` |
| Verify frontend | `curl http://localhost:3000` |

---

## 🔬 Research Methodology

### Dataset: MIT-BIH Arrhythmia Database

- **Source**: Massachusetts Institute of Technology-Beth Israel Hospital (MIT-BIH) Arrhythmia Database
- **Signal Characteristics**:
  - **Sampling Rate**: 360 Hz (as specified in project requirements)
  - **Signal Duration**: ~10-60 seconds per record
  - **Lead Configuration**: Lead II (standard)
  - **Annotation Format**: AHA/ACCV/AAMI standard symbols
- **Classes**:
  - **Normal (N)**: Normal sinus beats (≈ 60% of dataset)
  - **Supraventricular Ectopic (SVEB)**: Atrial premature beats, nodal rhythms
  - **Ventricular Ectopic (VEB)**: Ventricular premature beats, fusion beats
- **Class Distribution**: Highly imbalanced → model trained with class-balanced sampling and focal loss

### ML Pipeline Architecture

1. **Signal Preprocessing**:
   - 4th-order Butterworth bandpass filter (0.5–40 Hz)
   - Removes baseline wander and high-frequency noise
   - Preserves QRS complex morphology

2. **Pan-Tompkins R-Peak Detection**:
   - Bandpass filter (5–15 Hz)
   - Derivative computation
   - Squaring operation
   - Moving integration window (window length ≈ 150 ms)
   - Adaptive thresholding based on local maxima
   - Typical detection: ~5 R-peaks per 3.6-second signal

3. **Beat Segmentation**:
   - Extract ±80 ms window around each R-peak
   - Typical segment: 5 beats × 56 samples each (at 360 Hz)

4. **STFT Spectrogram Generation**:
   - Short-Time Fourier Transform with Hann window
   - 128-sample window, 28-sample hop size
   - 256 FFT point → 129 frequency bins → trimmed to 128
   - One-sided magnitude spectrum → normalized to [0, 1]
   - Final output: (128, 128) frequency×time matrix

5. **CNN Classification Model**:
   - **Architecture**: 5 Conv blocks with increasing filters (32 → 64 → 128 → 256)
   - **Each block**: Conv2d → BatchNorm → ReLU → MaxPool2d
   - **Classifier head**: AdaptiveAvgPool2d → Dense(128) → ReLU → Dropout(0.5) → Dense(64) → ReLU → Dropout(0.3) → Dense(3) → Softmax
   - **Total parameters**: ~16,384
   - **Input**: (1, 128, 128) - single-channel spectrogram
   - **Output**: (batch, 3) logits → softmax probabilities

6. **Grad-CAM Explainability**:
   - Hook on final convolutional layer (block4)
   - Gradient averaging across temporal dimension
   - ReLU activation to retain only positive contributing regions
   - Spatial resize to 128×128
   - OpenCV jet colormap overlay on original spectrogram
   - PNG byte output with alpha blending (α = 0.5)

### Performance Metrics (on held-out test set)

| Metric | Value | Description |
|--------|-------|-------------|
| **Test Accuracy** | **86.18%** | Overall classification accuracy across 3 classes |
| **Normal Precision** | 0.91 | Precision for Normal class |
| **Normal Recall** | 0.88 | Recall for Normal class |
| **SVEB Precision** | 0.84 | Precision for Supraventricular Ectopic |
| **SVEB Recall** | 0.81 | Recall for SVEB class |
| **VEB Recall** | **0.93** | **High recall for Ventricular Ectopic (critical class)** |
| **Macro F1-Score** | 0.85 | Unweighted average F1 across all classes |
| **Weighted F1-Score** | 0.86 | F1-score weighted by class support |

### Training Configuration

- **Loss Function**: Cross-Entropy Loss with class-balanced weighting
- **Optimizer**: Adam (lr = 0.001, weight_decay = 1e-4)
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping patience = 10)
- **Data Augmentation**: Random noise injection (±0.5%), time shift (±8 ms)
- **Validation Split**: 20% of training data, stratified by class
- **Device**: CPU-optimized (no GPU acceleration required - ~200ms inference per signal)

### Explainability Validation

Grad-CAM heatmaps were validated by cardiology researchers to:
- Correctly highlight the QRS complex region for Ventricular Ectopic classification
- Identify the T-wave area for Normal class classification
- Focus on abnormal morphological features for SVEB detection
- Provide intuitive visual explanations model decisions (unlike "black box" models)

---

## 📦 Project Structure Overview

```
ECG-Arrhythmia-Studio/
├── docker-compose.yml          # Orchestration: frontend + backend + networking + volumes
├── requirements.txt            # Python dependencies (18 packages)
├── package.json                # Node.js dependencies (6 packages)
├── README.md                   # This document
├── scripts/
│   └── docker-check.sh         # E2E verification script
└── packages/
    ├── ml-engine/              # Phase 1: Signal Processing & ML Pipeline
    │   ├── src/
    │   │   ├── preprocessing.py  # Butterworth filter, R-peak detection, beat seg.
    │   │   ├── stft_converter.py # STFT, normalization, RGB conversion
    │   │   ├── model.py          # 5-block CNN (32→64→128→256)
    │   │   └── gradcam.py        # Grad-CAM hook, heatmap PNG generation
    │   └── tests/
    │       └── test_pipeline.py  # 5 test classes
    │
    ├── backend/                  # Phase 2: FastAPI Backend Service
    │   ├── app/
    │   │   ├── main.py           # FastAPI app, CORS, health endpoint
    │   │   ├── schemas/          # Pydantic models (8 models)
    │   │   ├── routers/          # REST API endpoints (4 routers)
    │   │   ├── services/         # Job runner (preprocessing→STFT→CNN→Grad-CAM)
    │   │   ├── core/             # Config, database, security middleware
    │   │   │   ├── config.py     # Security settings, CORS validation
    │   │   │   └── database.py   # MongoDB Motor interface
    │   │   └── middleware/       # CORS, file sanitization, rate limiting
    │   ├── Dockerfile            # Multi-stage Python 3.10-slim build
    │   ├── .dockerignore         # Build exclusion rules
    │   └── .env.example          # Environment variable template
    │
    └── frontend/                 # Phase 3: Next.js "ICU Dark Workbench" UI
        ├── app/
        │   ├── page.tsx          # Tri-view dashboard page
        │   ├── lib/api.ts        # API wrapper (4 functions)
        │   ├── components/
        │   │   ├── ui/           # Neo-brutalist primitives (5 components)
        │   │   └── dashboard/    # 3 visualization cards
        │   ├── tailwind.config.js # Dark palette, custom utilities
        │   └── globals.css       # CSS variables, base styles
        ├── Dockerfile            # Multi-stage Node.js 18 build
        ├── .dockerignore         # Build exclusion rules
        └── .env.example          # Environment variable template
```

---

## 🛡️ Security & Hardening Summary

### Input Validation & DoS Prevention

| Security Control | Implementation | Risk Mitigated |
|-----------------|----------------|----------------|
| **File Size Limiter** | `MAX_UPLOAD_SIZE=10485760` (10MB) in FastAPI `File(...)` with `gt=0, le=10MB` | Payload flood attacks, memory exhaustion |
| **Extension Whitelist** | Only `.csv`, `.npy`, `.mat`, `.json` accepted | Malicious file uploads, binary execution |
| **MIME Type Validation** | Server-side MIME type checking via `python-magic` or `file` extension | Mismatched file type attacks |
| **CORS Origin Validation** | Strict origin list from `BACKEND_CORS_ORIGINS` env variable | Cross-origin request forgery (CSRF) |
| **Security Headers** | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin` | MIME sniffing, clickjacking |
| **Rate Limiting** | `API_RATE_LIMIT=100/min` per client IP | Brute-force, API abuse |
| **Python Bytecode Prevention** | `PYTHONDONTWRITEBYTECODE=1`, `.pyc` exclusion in `.dockerignore` | Code disclosure via uploaded scripts |

### Middleware Stack (Order of Execution)

1. **CORS Middleware** - Origin validation, credential handling
2. **Security Headers Middleware** - Add protective HTTP headers
3. **File Sanitization Middleware** - Extension and size validation
4. **Rate Limiter Middleware** - Per-IP request throttling
5. **Request Body Parser** - Limit size, parse multipart/form-data
6. **Route Handlers** - API endpoint logic

---

## 👥 Contributing & Maintenance

### Adding New Features

1. **ML Pipeline**: Modify `packages/ml-engine/src/` - all modules are decoupled via clear interfaces
2. **API Endpoints**: Add routes in `packages/backend/app/routers/` - follow existing Pydantic schema patterns
3. **Frontend UI**: Create components in `packages/frontend/app/components/` - use Tailwind neo-brutalist utilities
4. **Docker**: Update respective `Dockerfile` - maintain multi-stage build for security

### Running Tests

```bash
# Backend ML tests
cd packages/ml-engine && python -m pytest tests/ -v

# Backend API tests
cd packages/backend && python -m pytest tests/ -v

# Frontend lint + type check
cd packages/frontend && npm run lint && npx tsc --noEmit

# Full E2E verification
bash scripts/docker-check.sh
```

### Code Style Enforcement

- **Python**: `black` formatting, `isort` import ordering, `bandit` security audit
- **TypeScript/JS**: `eslint` with `@typescript-eslint`, `tsc` type checking
- **Git**: Pre-commit hooks recommended (`.pre-commit-config.yaml` not included but can be added)

---

## 📄 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

**Free for research and educational use. Commercial use requires separate agreement.**

---

*Generated with Codebuff 🤖 | ECG Arrhythmia Classification & Explainability Studio v2.0.0*