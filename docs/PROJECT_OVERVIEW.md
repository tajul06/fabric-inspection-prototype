# Fabric Inspection Prototype - Final Project Overview

## Purpose
This project delivers a working prototype for fabric defect inspection. It combines fabric classification, pattern classification, anomaly detection, and defect scoring into a single web app with a camera-assisted inspection workflow and summary dashboards.

## Audience
- Stakeholders: understand what the system does, what it produces, and how it supports inspection decisions.
- Developers: understand the system architecture, model stack, and how to run and extend the app.

## What It Does
- Classifies fabric type (knitted vs woven).
- Classifies pattern type (plain, stripe, floral) when possible.
- Detects and localizes anomalies with heatmaps.
- Suggests defect classes and confidence.
- Computes a 4-point score and decision (Accept/Hold/Reject).
- Supports few-shot inspection for unknown patterns (WinCLIP) and prototype-based anomaly classification.
- Stores inspection history and exposes dashboards.

## System Architecture
- Web app: Flask + SQLAlchemy
  - App factory: [app/__init__.py](app/__init__.py)
  - Configuration: [app/config.py](app/config.py)
  - Database models: [app/models.py](app/models.py)
  - Routes: [app/routes/main.py](app/routes/main.py), [app/routes/adapt.py](app/routes/adapt.py), [app/routes/profiles.py](app/routes/profiles.py)
- ML runtime: model loading, inference routing, and fallbacks in [app/ml_models.py](app/ml_models.py)
- UI: Tailwind + Alpine JS templates in [app/templates](app/templates) and scripts in [app/static/js/inspection.js](app/static/js/inspection.js)

## Core Data Flow
1. Upload image (or capture via camera) on the Inspection page.
2. Preprocess with CLAHE to normalize lighting.
3. Predict fabric type (or use manual override for WinCLIP path).
4. Predict pattern type (auto mode) or use few-shot WinCLIP (unknown mode).
5. Run anomaly detector:
   - Pattern-specific PatchCore when available.
   - Fabric-level PatchCore fallback.
   - Classic image-based fallback if checkpoints are missing.
6. Generate anomaly heatmap and blend view.
7. Compute 4-point score and decision.
8. Persist record and serve result assets.

## Model Stack
- Fabric classifier: ResNet-50 (binary knitted vs woven).
- Pattern classifier: EfficientNet (pattern classes per fabric type).
- Anomaly detection:
  - PatchCore checkpoints per fabric or pattern.
  - Fallback to heuristic anomaly map when PatchCore is unavailable.
- Few-shot detector: WinCLIP (k-shot reference images).
- Prototype-based anomaly classifier for knitted and woven support libraries.

Checkpoint discovery details and expected layouts are documented in [setup_instructions.md](setup_instructions.md).

## User Workflows
- Inspection (main flow): upload or capture, run inference, view heatmaps, bounding boxes, and decision.
- Few-shot Adaptation:
  - Detector flow: run WinCLIP with optional normal samples.
  - Classifier flow: add labeled support images and optionally test a query image.
- Profiles: manage reusable thresholds for future expansion.
- Dashboard: view inspection volume, averages, and recent results.

Templates for these pages live in:
- [app/templates/inspection.html](app/templates/inspection.html)
- [app/templates/adapt.html](app/templates/adapt.html)
- [app/templates/profiles.html](app/templates/profiles.html)
- [app/templates/dashboard.html](app/templates/dashboard.html)

## API Endpoints
- GET /: Inspection UI
- POST /: Run inspection
- POST /api/blend: Generate blended heatmap for a result id
- POST /api/local-score: Query anomaly score at a pixel location
- POST /api/pipeline-preview: Quick preview of the inference pipeline
- GET /adapt: Few-shot adaptation UI
- POST /adapt: Run detector or update support sets
- GET /profiles: Profiles UI
- GET /profiles/dashboard: Metrics dashboard

## Data and Outputs
- Uploads: [uploads](uploads)
- Processed outputs (images and anomaly maps): [processed](processed)
- SQLite database: [fabric_inspection.db](fabric_inspection.db)

Generated assets per inspection include:
- Original image, preprocessed image, heatmap, blended heatmap, bbox overlay
- Anomaly map stored as .npy

## Configuration
Key config values in [app/config.py](app/config.py):
- `UPLOAD_FOLDER`, `PROCESSED_FOLDER`, `MODEL_DIR`
- `SQLALCHEMY_DATABASE_URI` (defaults to local SQLite)
- `MAX_CONTENT_LENGTH` (25 MB)

## How To Run
Follow the setup guide in [setup_instructions.md](setup_instructions.md), then:

```bash
python run.py
```

App runs at http://127.0.0.1:5000

## Build Executables
- CPU build: [build_cpu.bat](build_cpu.bat)
- CUDA build: [build_cuda.bat](build_cuda.bat)
- PyInstaller spec: [main.spec](main.spec)

## Troubleshooting
- Missing checkpoints: the app falls back to heuristic anomaly maps. For best results, confirm model files exist per [setup_instructions.md](setup_instructions.md).
- WinCLIP path: requires at least 3, 5, or 10 support images. The UI enforces these shot counts.
- Camera capture: requires a browser with `getUserMedia` support and permission.
- Large files: uploads are capped at 25 MB (configurable in [app/config.py](app/config.py)).

## Known Gaps and Next Steps
- Automated tests are not included yet. Add unit tests for ML routing and utility functions.
- Consider exposing profile thresholds to the inspection flow for calibration.
- Add export utilities for reporting and audit logs.
