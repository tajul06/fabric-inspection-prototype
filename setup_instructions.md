# Fabric Inspection Prototype Setup

## 1. Environment

### CPU
1. Create a virtual environment.
2. Install dependencies:
   - `pip install -r requirements_cpu.txt`

### CUDA
1. Install CUDA-compatible PyTorch first if needed.
2. Install dependencies:
   - `pip install -r requirements_cuda.txt`

## 2. Model Files

The app now supports real checkpoint loading for your existing workspace artifacts.

Detected and supported formats:
- Fabric classifier: `.pt` checkpoint with `model_state_dict` and `class_to_idx`.
- PatchCore models: anomalib Lightning `.ckpt` files (`state_dict` + `hyper_parameters`).

Default checkpoint discovery order:
1. Local `models/` folder (for portable packaging)
2. Existing workspace training outputs:
   - `Fabric_classifier/restnet50/resnet50_run/best_model.pt`
   - `Anomaly_detector_knitten/anomalib_knitten/anomalib_patchcore_knitten/Patchcore/knitten_patchcore/v0/weights/lightning/model.ckpt`
   - `Anomaly_detector_woven/anomalib_patchcore_woven/anomalib_patchcore_woven/Patchcore/woven_patchcore/v0/weights/lightning/model.ckpt`

Optional portable local copies (inside `models/`):
- `fabric_classifier_resnet50.pt` or `fabric_classifier_resnet50.pth`
- `knitted_patchcore.ckpt`
- `woven_patchcore.ckpt`
- MVREC checkpoints (optional in prototype)

## 3. Run App

- `python run.py`

Open: `http://127.0.0.1:5000`

## 4. PatchCore + Heatmap Notes

This prototype follows anomalib PatchCore deployment patterns and anomaly map handling practices.

Official references:
- Anomalib repository: https://github.com/open-edge-platform/anomalib
- PatchCore implementation path: https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/patchcore
- Inferencing guidance: https://anomalib.readthedocs.io/en/latest/markdown/guides/how_to/inference/index.html

In this prototype:
- PatchCore is loaded from Lightning `.ckpt` through anomalib `Patchcore + Engine.predict`.
- PatchCore output is normalized to an anomaly map in `[0, 1]`.
- Heatmap is generated via OpenCV colormap and displayed separately.
- Blend mode overlays heatmap on the original image using OpenCV weighted blending.

## 5. Build Executables

- CPU build: `build_cpu.bat`
- CUDA build: `build_cuda.bat`

Both scripts generate output via `pyinstaller main.spec`.
