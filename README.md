# Fabric Inspection Prototype

A Flask-based web application for automated fabric defect detection, classification, and anomaly scoring using deep learning models. Supports both knitted and woven fabrics with real-time inspection feedback and live camera capture.

<img width="1426" height="1100" alt="fig_4_12_inspection_results" src="https://github.com/user-attachments/assets/b646d189-bc62-4086-92ec-52841acabff8" />

## Features

### Core Inspection Capabilities
- **Fabric Type Classification**: Automatic detection of knitted vs. woven fabrics using ResNet50.
- **Pattern Recognition**: 
  - Auto mode: Classifier-based pattern detection for structured fabrics.
  - Unknown mode: WinCLIP-based few-shot pattern learning with 3/5/10 support images.
- **Anomaly Detection**: 
  - PatchCore models for knitted and woven fabrics.
  - Produces anomaly score and per-pixel anomaly maps with heatmap visualization.
- **Anomaly Classification**:
  - Prototypical networks for fine-grained anomaly type classification.
  - Refines generic anomaly predictions with class-specific defect categorization.
- **Defect Analysis**: Connected component analysis for defect localization, bounding boxes, and severity classification.
- **Measurement Mode**: 
  - Auto Ratio: Calculates fabric dimensions based on aspect ratio.
  - Manual Area: User-specified width/height (cm) for precise area calculations.

### UI/UX
- **Interactive Inspection Dashboard**: Real-time heatmap blending with alpha transparency control.
- **Local Anomaly Probing**: Click on heatmap to inspect local anomaly scores and suggested defect types.
- **Live Camera Capture**: Built-in `getUserMedia` capture with multi-device support (USB cameras, DroidCam, etc.).
- **Support Image Management**: Capture multiple reference images directly in the browser for WinCLIP inference.
- **Result History**: View recent inspections in a persistent SQLite database.

### Image Processing
- **CLAHE Preprocessing**: Contrast-limited adaptive histogram equalization for enhanced feature detection.
- **Heatmap Blending**: Overlay anomaly heatmaps on original images with adjustable transparency.
- **Automatic Resizing**: Handles various input image sizes with proper aspect ratio preservation.
<img width="1364" height="1244" alt="fig_4_11_clahe_comparison" src="https://github.com/user-attachments/assets/8c23aa64-0090-45dc-bbff-33018cc24813" />

## Architecture

<img width="1024" height="1024" alt="figure_3_1_system_architecture" src="https://github.com/user-attachments/assets/dc8a8f9b-bb1c-42bb-981c-dd81c179a317" />



### Project Structure

```
fabric-inspection-prototype/
├── app/
│   ├── __init__.py              # Flask app factory
│   ├── config.py                # Configuration (paths, settings)
│   ├── ml_models.py             # ML inference runners (PatchCore, WinCLIP, ProtoClassifier)
│   ├── models.py                # SQLAlchemy database models
│   ├── utils.py                 # Image processing utilities
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── main.py              # Inspection routes, API endpoints
│   │   ├── profiles.py          # User profiles (if used)
│   │   └── adapt.py             # Adaptation routes (if used)
│   ├── templates/
│   │   ├── base.html            # Base template with navigation
│   │   ├── inspection.html      # Main inspection UI (with camera capture panel)
│   │   ├── dashboard.html       # Dashboard view
│   │   └── profiles.html        # Profiles view
│   └── static/
│       ├── js/inspection.js     # Alpine.js logic for camera, capture, heatmap interaction
│       ├── css/               # Global styles (Tailwind CDN)
│       └── images/            # Static assets
├── models/                      # Local portable model checkpoints (optional)
│   ├── fabric_classifier_resnet50.pt
│   ├── knitted_patchcore.ckpt
│   └── woven_patchcore.ckpt
├── uploads/                     # Temporary upload storage
├── processed/                   # Generated results (anomaly maps, heatmaps, blended)
├── Fabric_classifier/           # Original fabric classifier training outputs
├── Fabric_pattern_classifier/   # Pattern classifier models (.pth files)
├── knitten anomaly classifier/  # Original knitted anomaly detection models
├── woven anomaly classifier/    # Original woven anomaly detection models
│   └── support/                 # Per-class reference images for prototypical learning
├── Pattern_based_Anomaly_Detect/ # Pattern-specific anomaly models
├── indiv anomaly detector/      # Individual per-pattern anomaly detectors
├── run.py                       # Entry point (Flask development server)
├── main.spec                    # PyInstaller spec for executable building
├── build_cpu.bat                # Build script for CPU executables
├── build_cuda.bat               # Build script for CUDA executables
├── requirements_cpu.txt         # CPU dependencies (pip freeze)
├── requirement_cpu1.txt         # Pipreqs-generated CPU dependencies
├── requirements_cuda.txt        # CUDA dependencies
└── fabric_inspection.db         # SQLite database (results, history)
```

### Data Flow
<img width="1568" height="2666" alt="fig_4_2_pipeline_flowchart" src="https://github.com/user-attachments/assets/259ff229-d001-4b19-b7c0-a2ff67436da0" />

1. **Input**: User uploads image or captures via live camera.
2. **Preprocessing**: CLAHE contrast enhancement applied.
3. **Classification**: 
   - Fabric type (knitted/woven) → ResNet50.
   - Pattern type → Classifier (auto) or WinCLIP (unknown).
4. **Anomaly Detection**: PatchCore model for detected fabric type → anomaly score + map.
5. **Prototypical Classification**: If prototypical network enabled for fabric type:
   - Extract region of interest from anomaly map (or full image if no significant anomaly).
   - Embed via ResNet18 backbone → Euclidean distance to class prototypes.
   - Refine anomaly class prediction with prototypical network output.
6. **Defect Analysis**: Connected components, bounding boxes, severity scoring.
7. **Measurement**: Area calculation (auto-ratio or manual).
8. **Decision**: 4-point scale (Pass/Hold/Reject) based on anomaly score + defect count/severity.
9. **Output**: Result stored in database; images saved to `processed/`.

## ML Models

### 1. Fabric Type Classifier (ResNet50)
- **Purpose**: Classify input as knitted or woven.
- **Path**: `Fabric_classifier/restnet50/resnet50_run/best_model.pt` or portable `models/fabric_classifier_resnet50.pt`.
- **Output**: Class + confidence.
  

### 2. Pattern Classifier (EfficientNet)
- **Purpose**: Classify fabric pattern (e.g., plain, stripe, print, plaid).
- **Models**: 
  - Knitted: `Fabric_pattern_classifier/best_efficientnet_fabric_print_knitten.pth`
  - Woven: `Fabric_pattern_classifier/best_efficientnet_fabric_print_woven.pth`
- **Used in**: Auto pattern mode.

### 3. PatchCore Anomaly Detection (anomalib)
- **Purpose**: Detect anomalies (defects, stains, tears) at pixel level.
- **Models**:
  - Knitted: `knitten anomaly classifier/best_proto_fabric_model.pth` or anomalib PatchCore `.ckpt`.
  - Woven: Anomalib PatchCore `.ckpt` from workspace.
- **Output**: Anomaly score (0–1) + anomaly map (same resolution as input).
- **Framework**: anomalib 2.0.0 with PyTorch backend.

### 4. WinCLIP Few-Shot Anomaly (anomalib)
- **Purpose**: Detect anomalies using user-provided reference images (few-shot learning).
- **Used in**: Unknown pattern mode (e.g., custom or unfamiliar fabric patterns).
- **Support Images**: 3, 5, or 10 reference images of normal/good fabric.
- **Output**: Anomaly score + map, specific to the provided reference set.
- **Framework**: anomalib WinClipModel with CLIP backbone.

### 5. Prototypical Networks (Knitted & Woven)
- **Purpose**: Few-shot anomaly classification for knitted and woven fabrics using prototypical networks.
- **Backbone**: ResNet18 for embedding extraction (224×224 input).
- **Architecture**:
  - Support images organized in class folders (e.g., `support/normal/`, `support/defect_type_1/`).
  - Class prototypes computed as mean embedding across all support images for that class.
  - Query image embedded and classified via Euclidean distance to prototypes.
  - Softmax applied to distances to produce class probabilities.
- **Support Paths**:
  - Knitted: `knitten anomaly classifier/support/` (class subdirectories with reference images).
  - Woven: `woven anomaly classifier/support/` (class subdirectories with reference images).
- **Model Checkpoints**:
  - Knitted: `knitten anomaly classifier/best_proto_fabric_model.pth`.
  - Woven: `woven anomaly classifier/best_proto_fabric_model.pth` (if available).
- **Dynamic Support Reloading**: Support images are monitored for changes; prototypes automatically recomputed if new reference images added/removed.
- **Integration**: Used as secondary classifier after anomaly detection; refines anomaly class predictions based on fine-grained fabric characteristics.
- **Output**: Class label (e.g., "normal", "defect_type_1") + confidence (0–1).
## Results

### Pattern Classification
| Knitted (2 classes) | Woven (23 classes) |
|---|---|
| <img width="1396" height="1160" alt="fig_4_7a_knitted_cm" src="https://github.com/user-attachments/assets/04ef108d-653a-411c-9d32-288c9d9f9135" /> | <img width="4161" height="3561" alt="fig_4_7b_woven_cm" src="https://github.com/user-attachments/assets/08c08199-270e-4094-acbf-d91918460855" />|

### Anomaly Detection Performance

Per-pattern PatchCore evaluation across all 22 woven patterns achieved a mean 
image-level AUROC of **96.8%**, substantially outperforming the fabric-level 
baseline of 86.92%. 18 of 22 patterns scored 98–100% AUROC; the remaining 
four (mostly floral prints) ranged 80.5–94.7%, likely due to higher intra-class 
visual variance.

<img width="2963" height="2662" alt="fig_4_8_woven_patchcore" src="https://github.com/user-attachments/assets/34e10df9-644b-4366-b2c2-1330b4df6ee5" />

### Defect Type Classification

Using Prototypical Networks (ResNet-18 backbone) to classify defect type once 
an anomaly is localized:

| Woven (7 classes, TILDA) | Knitted (5 classes, ISL-Knit) |
|---|---|
| <img width="2184" height="1755" alt="fig_4_10a_proto_woven_cm" src="https://github.com/user-attachments/assets/9e3c5a87-2ee8-4e58-a659-d10121c8db54" /> | <img width="1905" height="1456" alt="fig_4_10b_proto_knitted_cm" src="https://github.com/user-attachments/assets/8df217a4-a4c9-40aa-899a-dbcf64bb748b" /> |
| **94.44%** val accuracy | **78.20%** val accuracy |

Woven defect classification performs near-perfectly across all 7 classes. 
Knitted defect classification is weaker (78.2%), with most confusion between 
visually similar classes — Snag/Stain and Hole/Thread Defect.

### Decision Engine — Severity Scoring
<img width="1568" height="1654" alt="fig_4_3_decision_tree" src="https://github.com/user-attachments/assets/6a7a30ff-fd67-4b93-94c4-3619f5a38a5d" />

## Setup

### Prerequisites
- Python 3.10 or later.
- (Optional) CUDA 11.8+ for GPU inference; otherwise CPU is supported.
- Git (for cloning if needed).

### Installation

#### 1. Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Unix/Linux/macOS
```

#### 2. Install Dependencies
**CPU:**
```bash
pip install --upgrade pip
pip install -r requirements_cpu.txt
```

**CUDA:**
```bash
# Install CUDA-compatible PyTorch first (if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_cuda.txt
```

#### 3. Verify Model Availability
The app auto-discovers models in this priority order:
1. Local `models/` folder (portable).
2. Workspace training outputs (original checkpoint paths).

Optional: Copy checkpoints to `models/` for portable packaging:
- `fabric_classifier_resnet50.pt` or `.pth`
- `knitted_patchcore.ckpt`
- `woven_patchcore.ckpt`

### Configuration
Edit `app/config.py` if needed:
- `SECRET_KEY`: Flask session secret (change for production).
- `UPLOAD_FOLDER`: Temp upload directory.
- `PROCESSED_FOLDER`: Results output directory.
- `MODEL_DIR`: Model checkpoint search path.
- `MAX_CONTENT_LENGTH`: Max upload size (default 25 MB).

## Dataset

Trained and evaluated across five fabric datasets covering woven and knitted 
defect/pattern samples (AITEX, TILDA, ZJU-Leaper, TFD, ISL-Knit).

<img width="990" height="1918" alt="fig_4_1_dataset_samples" src="https://github.com/user-attachments/assets/1780e45e-93df-4f33-8bc3-5c1f1c14f5d3" />


## Running the Application

<img width="1920" height="1080" alt="figure_3_6_webapp_ui (a)" src="https://github.com/user-attachments/assets/752f410a-0502-4c41-9641-864ab553c076" />
<img width="1920" height="1080" alt="figure_3_6_webapp_ui (b)" src="https://github.com/user-attachments/assets/d132cd07-6057-4994-a618-19386bd8784b" />
<img width="1920" height="1080" alt="figure_3_6_webapp_ui (c)" src="https://github.com/user-attachments/assets/9b7b3ffc-0ce8-447e-bd7a-cdf744c311c8" />



### Development Server
```bash
python run.py
```
Opens at `http://127.0.0.1:5000`

### Access Remotely
- Find your machine's LAN IP: `ipconfig` (Windows) or `ifconfig` (Unix).
- Access from another device on the same network: `http://<LAN_IP>:5000`
- Ensure firewall allows port 5000 traffic.

### Build Executables
**CPU:**
```bash
build_cpu.bat
# Output: dist/main.exe
```

**CUDA:**
```bash
build_cuda.bat
# Output: dist/main.exe
```

*Note: Executable size is large due to bundled PyTorch models (~1.5–2 GB).*

## Usage Guide

### Basic Workflow

1. **Navigate to Inspection**: Home page → "Inspection" tab.
2. **Upload Image**:
   - Manual: Click "Upload Fabric Image" input.
   - Live Camera: 
     - Click "Start" to initialize camera.
     - Select device from dropdown (e.g., DroidCam).
     - Click "Capture Main Image" to auto-fill and submit.
3. **Configure Detection**:
   - **Pattern Mode**: 
     - Auto: Use trained classifier.
     - Unknown: Use WinCLIP with support images.
   - **Fabric Type**: Override auto-detection if needed.
   - **Support Shots** (Unknown mode): 3, 5, or 10 reference images.
   - **Measurement Mode**: Auto-ratio or manual area.
4. **Run Inference**: Click "Inspect" → Processing begins.
5. **Review Results**:
   - **Original & Preprocessed**: Input images.
   - **Defect Bounding Boxes**: Detected defects.
   - **Anomaly Heatmap**: 
     - Click to probe local anomaly score.
     - Toggle "Blend Heatmap on Original" and adjust transparency.
   - **Inference Summary**: Scores, decision (Accept/Hold/Reject), defect details.

### Live Camera Capture

**Setup DroidCam (Mobile):**
1. Install DroidCam app on phone.
2. Connect phone to same Wi-Fi network as laptop.
3. In DroidCam app, note the **IP:Port** shown.
4. On laptop, install DroidCam client (optional, or use browser driver).
5. In browser, camera dropdown will list "DroidCam" once connected.

**Capture Workflow:**
- Start camera → Select DroidCam → Capture Main Image (auto-submits).
- For Unknown pattern mode: Capture Support Image multiple times → Inspect (manual submit).

### WinCLIP Few-Shot Mode

**Scenario**: You have a custom fabric pattern not in the classifier.

1. Set Pattern Mode → "Unknown".
2. Set Support Shots → 5 (or 3/10).
3. Click "Capture Support Images" (or upload) → Add 5 good/normal reference images.
4. Upload main inspection image.
5. Click "Inspect" → WinCLIP compares main image against the 5 reference images.
6. Result: Anomaly score tailored to your fabric's reference set.

### Prototypical Network Classification

**Overview**: The prototypical network provides fine-grained anomaly classification by learning from a few reference images per defect class.

**Setup:**
1. Organize support images in folder structure:
   - `knitten anomaly classifier/support/normal/` → Images of normal knitted fabric.
   - `knitten anomaly classifier/support/defect_type_1/` → Images of specific defect type.
   - `knitten anomaly classifier/support/defect_type_2/` → Additional defect types.
   - (Same structure for woven in `woven anomaly classifier/support/`)
2. Place 5–20 reference images per class (more examples = better prototypes).
3. Restart server; prototypical classifiers load automatically if checkpoints available.

**How It Works:**
1. ResNet18 embeds all support images → Compute class prototype (mean embedding).
2. During inference, anomalous image region extracted and embedded.
3. Euclidean distance computed to each class prototype.
4. Class with smallest distance selected; softmax produces confidence.
5. Anomaly class prediction refined from generic to specific.

**Dynamic Updates:**
- Support image folders monitored for changes.
- New/removed images automatically trigger prototype recomputation.
- No server restart needed for support image updates.

**Result Routing:**
- Routes track which model provided final classification: e.g., `patchcore:knitted_proto_crop` = PatchCore detected anomaly, knitted proto classifier refined class using cropped region.

## API Endpoints

### Main Routes
- **`GET /`**: Dashboard (home page).
- **`GET /inspection`**: Inspection form page.
- **`POST /inspection`**: Submit inspection (multipart form).
- **`GET /result_file/<filename>`**: Download processed images.

### API Endpoints
- **`POST /blend_api`**: Generate blended heatmap with custom alpha.
- **`POST /local_score_api`**: Get local anomaly score at clicked pixel.
- **`POST /pipeline_preview_api`**: Preview preprocessing pipeline.

## Troubleshooting

### "Camera access not supported"
- Browser security: Camera access requires HTTPS in production or localhost.
- Permissions: Grant camera access when browser prompts.
- Device: Ensure camera device is available and not in use.

### "Camera preview shows green/colored noise"
- Codec issue: Try different camera device or browser.
- DroidCam: Switch to MJPEG mode in DroidCam settings.
- Browser: Disable hardware acceleration or try Firefox.

### "No WinCLIP model found for knitted"
- Issue: WinCLIP is shared across fabrics; ensure anomalib[full] is installed.
- Fix: `pip install anomalib[full]` or check `anomalib` version.

### Inference is slow
- GPU not detected: Check if CUDA is installed and PyTorch recognizes it (`torch.cuda.is_available()`).
- Model loading: First inference takes longer due to model loading to VRAM.
- CPU mode: Switch to CUDA for 2–5x speedup.

### "Operands could not be broadcast together"
- Cause: Image size mismatch between heatmap and input.
- Fix: Automatic resizing is applied; restart server if errors persist.

## Dependencies

### Core
- **Flask** 3.1.3: Web framework.
- **Flask-SQLAlchemy** 3.1.1: Database ORM.
- **SQLAlchemy** 2.0.49: Database toolkit.

### ML & Image Processing
- **PyTorch** 2.6.0 + **torchvision** 0.21.0: Deep learning framework.
- **anomalib** 2.0.0 (with [full] extras recommended): Anomaly detection models.
- **timm** 1.0.26: Pre-trained model hub (EfficientNet, ResNet).
- **OpenCV** 4.11.0.86: Image processing and visualization.
- **NumPy** 2.4.4: Numerical computing.

### Web
- **Werkzeug** 3.1.8: WSGI utilities.
- **Requests** 2.33.1: HTTP client.

### Frontend
- **Alpine.js**: Reactive UI (via CDN, not in requirements).
- **Tailwind CSS**: Styling (via CDN, not in requirements).

## Performance Tips

1. **GPU Acceleration**: Use CUDA if available for 2–5x speedup.
2. **Image Size**: Smaller inputs (e.g., 512×512) are faster but may lose detail.
3. **Model Caching**: Models are cached after first load; subsequent inferences are faster.
4. **Batch Processing**: Current app processes one image at a time; multi-image batching could improve throughput.

## Development & Customization

### Adding a New Anomaly Model
1. Place `.ckpt` or `.pt` checkpoint in `models/` folder.
2. Update `ml_models.py` model discovery logic.
3. Restart server.

### Extending Prototypical Networks

**Add New Defect Classes:**
1. Create subdirectories in support folder:
   - `knitten anomaly classifier/support/new_defect_class/` or `woven anomaly classifier/support/new_defect_class/`
2. Add 5–20 reference images per class (PNG/JPG format).
3. Server automatically detects and reloads prototypes on next inference.
4. New class predictions appear in inspection results.

**Train Custom Prototypical Backbone:**
- Current implementation uses pre-trained ResNet18.
- To fine-tune or train custom backbone:
  1. Prepare labeled dataset with defect class folders.
  2. Train ResNet18 variant using existing `best_proto_fabric_model.pth` checkpoint.
  3. Save trained weights to `knitten anomaly classifier/best_proto_fabric_model.pth` or woven equivalent.
  4. Server auto-loads updated checkpoint on restart.

### Customizing the UI
- Edit `app/templates/inspection.html` for layout/controls.
- Edit `app/static/js/inspection.js` for Alpine.js state/logic.
- Tailwind CSS classes are applied inline; update via CDN or local build.

### Extending the Database
- Add new columns to `app/models.py` (InspectionResult, etc.).
- Run database migration or delete `fabric_inspection.db` to recreate schema.

## License & References

### Anomaly Detection
- **PatchCore**: https://github.com/open-edge-platform/anomalib
- **WinCLIP**: anomalib WinClipModel (built-in).
- Docs: https://anomalib.readthedocs.io/

### Few-Shot Learning & Prototypical Networks
- **Easy-FSL**: https://github.com/sicara/easy-few-shot-learning
- **Prototypical Networks**: Snell et al., "Prototypical Networks for Few-shot Learning" (NIPS 2017).

### Model Architectures
- **ResNet50**: torchvision.models
- **EfficientNet**: timm library
- **CLIP**: OpenAI, integrated in anomalib.

## Notes

- This is a **prototype** for research and development; production deployment requires security hardening (HTTPS, authentication, input validation).
- Model paths are auto-discovered; ensure correct folder structure or place portable copies in `models/`.
- Database is SQLite; for production, switch to PostgreSQL or MySQL in `config.py`.
- Live camera works on localhost and LAN; HTTPS required for remote HTTPS connections.

## Support & Feedback

For issues, questions, or feature requests, refer to the project's issue tracker or documentation.
