# Installation Guide

## Prerequisites

Before installing the Fabric Defect Detection & Inspection framework, ensure you have:

- **Python 3.10 or later** (3.11 recommended)
- **pip** (Python package manager)
- **Git** (for version control)
- **4GB+ RAM** (8GB+ recommended)
- **GPU** (optional but strongly recommended for performance)
  - NVIDIA GPU with CUDA 11.8+ support
  - 2GB+ VRAM for models

## System Requirements

### Minimum
- CPU: Intel i5/AMD Ryzen 5 or equivalent
- RAM: 4GB
- Storage: 10GB (for models and data)
- OS: Windows 10/11, Linux, or macOS

### Recommended
- CPU: Intel i7/AMD Ryzen 7 or equivalent
- RAM: 8GB+
- Storage: 20GB (SSD recommended)
- GPU: NVIDIA RTX 2070+ with 4GB VRAM
- OS: Windows 10/11 or Ubuntu 20.04+

## Installation Steps

### Step 1: Clone or Download Repository

**Option A: Using Git (Recommended)**
```bash
git clone https://github.com/tajul06/fabric-inspection-prototype.git
cd fabric-inspection-prototype
```

**Option B: Download ZIP**
- Download from GitHub
- Extract to desired location
- Open terminal in the extracted folder

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Upgrade pip

```bash
# Windows
python -m pip install --upgrade pip

# Linux/macOS
python3 -m pip install --upgrade pip
```

### Step 4: Install Dependencies

#### For CPU Only

```bash
pip install -r requirements_cpu.txt
```

This is suitable for:
- Testing and development
- Limited inference (slower processing)
- Machines without NVIDIA GPU

#### For GPU (CUDA 11.8+)

```bash
# Step 1: Install CUDA-compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Step 2: Install remaining dependencies
pip install -r requirements_cuda.txt
```

**Verify CUDA installation:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 5: Verify Installation

```bash
# Check Python version
python --version  # Should be 3.10+

# Check main dependencies
python -c "import torch; import flask; import cv2; print('All imports successful!')"

# List installed packages
pip list
```

## Post-Installation Setup

### Create Required Directories

```bash
mkdir uploads      # For uploaded images
mkdir processed    # For result outputs
mkdir cache        # For temporary files
```

### Verify Model Files

The application expects models in these locations:

```
models/
├── fabric_classifier_resnet50.pt
├── knitted_patchcore.ckpt
└── woven_patchcore.ckpt

Fabric_pattern_classifier/
├── best_efficientnet_fabric_print_knitten.pth
└── best_efficientnet_fabric_print_woven.pth
```

**Note**: Models should already be included. If missing, download from:
- Project releases page
- Google Drive (if provided)
- Train your own using `train.py`

### Configure Application

Edit `app/config.py` if needed:

```python
# Example configuration
SECRET_KEY = 'your-secret-key-here'  # Change for production
DEBUG = False                         # Set to False for production
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25MB max file size
CUDA_AVAILABLE = True  # Set to False for CPU-only mode
```

## Running the Application

### Development Server

```bash
python run.py
```

Output should show:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

Access the application at: **http://localhost:5000**

### Production Deployment

For production use, use a WSGI server:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 run:app
```

### Access from Other Devices

1. Find your machine's IP:
   ```bash
   # Windows
   ipconfig
   
   # Linux/macOS
   ifconfig
   ```

2. Access from another device:
   ```
   http://<your-ip>:5000
   ```

3. Ensure firewall allows port 5000:
   - Windows: Check Windows Defender Firewall settings
   - Linux: `sudo ufw allow 5000`

## Building Executables

### Windows Executable (CPU)

```bash
build_cpu.bat
```

Output: `dist/main.exe` (~1.5GB)

### Windows Executable (CUDA)

```bash
build_cuda.bat
```

Output: `dist/main.exe` (~2GB)

### Manual Build

```bash
pip install pyinstaller
pyinstaller main.spec
```

## Troubleshooting Installation

### Import Error: "No module named 'torch'"

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# Reinstall PyTorch
pip install torch torchvision torchaudio
```

### CUDA Not Available

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Port Already in Use

```bash
# Windows - Find and kill process on port 5000
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/macOS
lsof -ti:5000 | xargs kill -9
```

### Out of Memory Error

```bash
# Reduce batch size in app/config.py
BATCH_SIZE = 1

# Or use CPU mode
CUDA_AVAILABLE = False
```

### Missing Dependencies

```bash
# Reinstall all dependencies
pip install -r requirements_cpu.txt --upgrade

# Or for GPU
pip install -r requirements_cuda.txt --upgrade
```

## Verification Checklist

After installation, verify:

- [ ] Python 3.10+ installed
- [ ] Virtual environment activated
- [ ] All dependencies installed without errors
- [ ] CUDA available (if GPU setup)
- [ ] Model files present
- [ ] Upload/processed directories created
- [ ] Application starts without errors
- [ ] Web interface accessible at localhost:5000

## What's Next?

- Read [Quick Start](Quick-Start) guide
- Explore [Web Interface](Web-Interface) guide
- Check [Configuration](Configuration) documentation
- Review [Troubleshooting](Troubleshooting) guide if issues arise

---

**Last Updated**: May 2026
