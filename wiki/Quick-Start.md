# Quick Start

Get the Fabric Defect Detection system up and running in 5 minutes!

## Prerequisites

- Python 3.10+
- Virtual environment (optional but recommended)
- 5-10 GB free disk space

## Installation (2 minutes)

### 1. Clone Repository
```bash
git clone https://github.com/tajul06/fabric-inspection-prototype.git
cd fabric-inspection-prototype
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/macOS
```

### 3. Install Dependencies
```bash
# For CPU
pip install -r requirements_cpu.txt

# For GPU (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_cuda.txt
```

## Running the Application (1 minute)

```bash
python run.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
```

**Open in browser:** http://localhost:5000

## Your First Inspection (2 minutes)

### Step 1: Navigate to Inspection
- Click **"Inspection"** in the navigation menu

### Step 2: Upload an Image

**Option A: Upload File**
- Click "Upload Fabric Image" button
- Select a JPG or PNG image of fabric
- Click "Open"

**Option B: Use Sample**
- The application includes sample fabric images
- Located in `knowledge/screenshots/`

### Step 3: Run Inspection
1. Leave settings at defaults for first test
2. Click **"Inspect"** button
3. Wait for processing (~2-5 seconds)

### Step 4: View Results

You'll see:
- **Original Image**: Input fabric photo
- **Defect Bounding Boxes**: Detected defect locations
- **Anomaly Heatmap**: Red areas indicate anomalies
- **Summary**: Pass/Hold/Reject decision with confidence

## Next Steps

### Learn More
- [Installation Guide](Installation-Guide) - Detailed setup
- [Web Interface Guide](Web-Interface) - UI walkthrough
- [Model Architecture](Model-Architecture) - Technical details
- [Few-Shot Learning](Few-Shot-Learning) - Concepts explained

### Try Advanced Features

**Live Camera Setup**
1. Connect camera (USB camera, phone via DroidCam)
2. In Inspection tab, click "Start Camera"
3. Select device from dropdown
4. Click "Capture Main Image"

**Unknown Pattern Mode**
1. Set Pattern Mode → "Unknown"
2. Capture 5 reference images of normal fabric
3. Upload inspection image
4. Click "Inspect" (uses few-shot learning)

### Build Standalone App
```bash
# Create Windows executable
build_cpu.bat  # CPU version
build_cuda.bat  # GPU version

# Run from dist/
dist/main.exe
```

## Common Commands

```bash
# Start development server
python run.py

# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# List installed packages
pip list

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Deactivate virtual environment
deactivate

# Update dependencies
pip install -r requirements_cpu.txt --upgrade
```

## Troubleshooting

**"ModuleNotFoundError: No module named 'flask'"**
```bash
# Make sure virtual environment is activated
.venv\Scripts\activate
pip install -r requirements_cpu.txt
```

**"Port 5000 already in use"**
```bash
# Kill existing process or use different port
# Edit app/config.py and change PORT = 5001
```

**"CUDA out of memory"**
```bash
# Use CPU mode - Set in app/config.py
CUDA_AVAILABLE = False
```

See [Troubleshooting Guide](Troubleshooting) for more help.

## Project Structure

```
fabric-inspection-prototype/
├── app/                  # Flask application
│   ├── ml_models.py     # Model inference
│   ├── routes/          # API endpoints
│   ├── templates/       # HTML pages
│   └── static/          # CSS, JS, images
├── models/              # Pre-trained models
├── uploads/             # User uploads
├── processed/           # Results
├── run.py               # Start server here
└── requirements_cpu.txt # Dependencies
```

## Key Features

✅ **Real-time Fabric Inspection** - Web-based UI  
✅ **Multi-Fabric Support** - Woven, knitted, custom  
✅ **Few-Shot Learning** - Adapt to new patterns with 5 examples  
✅ **Live Camera Capture** - Inspect directly from camera  
✅ **Anomaly Detection** - Visual heatmaps of defects  
✅ **Detailed Analytics** - Defect counting and severity  
✅ **GPU Acceleration** - Fast inference on NVIDIA GPUs  

## Performance

| Task | Time |
|------|------|
| Single Image Inference | 200-500ms |
| Full Pipeline | 400-700ms |
| Camera Preview | Real-time |
| Model Loading | 1-2 seconds |

## Keyboard Shortcuts

- `Esc` - Close modals
- `Enter` - Submit forms
- `Space` - Pause/Resume video

## API Overview

Main endpoints:

```
GET  /                       # Dashboard
GET  /inspection             # Inspection page
POST /inspection             # Process image
POST /blend_api              # Blend heatmap
POST /local_score_api        # Local anomaly score
```

For full API docs, see [API Reference](API-Reference).

---

**Stuck?** Check [Troubleshooting Guide](Troubleshooting) or create a GitHub issue!

**Last Updated**: May 2026
