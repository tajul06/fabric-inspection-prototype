# Web Interface Guide

Complete walkthrough of the Fabric Inspection web application interface.

## Overview

The web application provides an intuitive dashboard for fabric inspection with real-time processing, live camera support, and detailed defect analysis.

## Main Dashboard

### Navigation

The top navigation bar includes:

- **Home** - Dashboard view with statistics
- **Inspection** - Fabric defect detection interface
- **Results** - View inspection history
- **About** - Project information

### Dashboard Metrics

Shows key statistics:
- **Total Inspections** - Images processed today
- **Average Anomaly Score** - Mean defect detection score
- **Pass Rate** - Percentage of accepted fabrics
- **Processing Time** - Average inference time

---

## Inspection Interface

### Image Upload Section

#### Upload Methods

**1. File Upload**
```
1. Click "Upload Fabric Image" button
2. Select JPG or PNG file from computer
3. Click "Open" to load image
4. Preview displays in "Original Image" box
```

**2. Live Camera Capture**
```
1. Click "Start Camera" button
2. Allow browser camera access when prompted
3. Select camera device from dropdown
4. Click "Capture Main Image" to photograph
5. Image automatically fills upload field
```

**Supported Formats**:
- JPEG (.jpg, .jpeg)
- PNG (.png)
- Maximum size: 25 MB
- Recommended: 1024×1024 or higher

---

### Configuration Panel

#### Fabric Type Selection
```
Options:
- Auto Detect (recommended) - System identifies fabric type
- Woven - Manual selection for woven fabrics
- Knitted - Manual selection for knitted fabrics
```

#### Pattern Mode

**Auto Mode** (Default)
- Uses trained pattern classifier
- Works for common patterns: plain, stripe, plaid, floral, print
- Fast inference (~100ms)
- Recommended for standard fabrics

**Unknown Mode** (Few-Shot)
- Uses WinCLIP with support images
- Works for custom/unfamiliar patterns
- Requires 3, 5, or 10 reference images
- Better for novel fabric types

**Support Shots Selection** (Unknown Mode only)
```
Options:
- 3 shots - Minimum, fastest, lower accuracy
- 5 shots - Balanced performance/accuracy (recommended)
- 10 shots - Maximum, slower, highest accuracy
```

#### Measurement Mode

**Auto Ratio**
- Calculates dimensions from aspect ratio
- Requires reference measurement in settings
- Fast, suitable for real-time quality control

**Manual Area**
- User-specified width and height in cm
- Most accurate for precise area calculations
- Recommended for detailed defect analysis

```
Example:
- Fabric Width: 150 cm
- Fabric Height: 200 cm
```

---

### Support Images (Unknown Mode)

**Capture Support Images**
1. Click "Capture Support Images" button
2. Select number of shots (3, 5, or 10)
3. For each shot:
   - Take photo of normal (defect-free) fabric
   - Vary angle, lighting, position
   - Click "Capture" for each image
4. Click "Confirm" when complete

**Best Practices**
- Use high-quality, well-lit images
- Include different fabric areas
- Capture from multiple angles
- Avoid heavy shadows or glare
- Use consistent background

---

## Results Display

### Original Image
Shows the uploaded fabric image with no processing.

### Preprocessed Image
Displays CLAHE (Contrast-Limited Adaptive Histogram Equalization) enhanced version used for analysis.

### Defect Detection

**Bounding Boxes**
- Red boxes mark detected defects
- Box size indicates defect extent
- Hover for defect details

**Anomaly Heatmap**
```
Color Meaning:
- Blue (0.0)     - Normal, no anomaly
- Green (0.3)    - Low anomaly probability
- Yellow (0.6)   - Medium anomaly
- Orange (0.8)   - High anomaly
- Red (1.0)      - Very high anomaly, likely defect
```

**Heatmap Blending**
1. Toggle "Blend Heatmap on Original"
2. Use slider to adjust transparency (0-100%)
3. 0% - Full original image
4. 50% - Equal blend
5. 100% - Full heatmap

**Local Anomaly Probing**
1. Enable "Probe Mode"
2. Click on heatmap to inspect
3. Shows local anomaly score (0-1)
4. Displays suggested defect type
5. Indicates severity level

---

## Inspection Results

### Decision Output

```
PASS ✓
├─ Anomaly Score: < 0.3
├─ Defects Found: 0-1
└─ Confidence: > 90%

HOLD ⚠
├─ Anomaly Score: 0.3-0.7
├─ Defects Found: 2-5
└─ Confidence: 70-90%

REJECT ✗
├─ Anomaly Score: > 0.7
├─ Defects Found: > 5
└─ Confidence: < 70%
```

### Defect Summary Table

| Field | Description |
|-------|-------------|
| ID | Defect identifier |
| Type | Hole, stain, tear, pattern break, etc. |
| Area (cm²) | Calculated defect size |
| Severity | Low, Medium, High |
| Location | Position on fabric (quadrant) |
| Confidence | Detection confidence 0-100% |

### Inference Statistics

```
Processing Breakdown:
├─ Preprocessing: 50ms
├─ Fabric Classification: 60ms
├─ Pattern Detection: 100ms
├─ Anomaly Detection: 150ms
├─ Defect Analysis: 40ms
└─ Total: 400ms
```

---

## History & Results

### Results List
- Shows recent inspections with thumbnails
- Click to view full details
- Filter by date, decision, fabric type
- Search by image name

### Result Details
- Original image
- All processed outputs
- Full defect analysis
- Scores and metrics
- Export options

### Export Options
```
- PNG: Annotated image with bounding boxes
- JPG: Compressed image for archival
- PDF: Full report with metrics
- CSV: Defect data for spreadsheet
- JSON: Complete analysis in JSON format
```

---

## Settings & Configuration

### General Settings
```
- Theme: Light/Dark mode
- Language: English, others (if available)
- Auto-save results: Yes/No
- Result retention: 7/30/90 days
```

### Model Settings
```
- Fabric Classifier: Enable/Disable
- Pattern Classifier: Enable/Disable
- Anomaly Detection: Enable/Disable
- Anomaly Threshold: 0.0-1.0 (default 0.5)
```

### Camera Settings
```
- Default Camera: Select preferred device
- Camera Resolution: 720p/1080p/4K
- Frame Rate: 24/30/60 fps
- Auto-focus: On/Off
```

### Performance Settings
```
- GPU Acceleration: On/Off
- Batch Size: 1-4 (GPU dependent)
- Image Resize: Auto/Manual
- Quality Mode: Fast/Balanced/High-Quality
```

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Escape` | Close modal/dialog |
| `Enter` | Submit form |
| `Ctrl+U` | Upload new image |
| `Ctrl+I` | Start inspection |
| `Ctrl+C` | Capture camera |
| `Ctrl+H` | Show help |
| `Ctrl+S` | Save results |
| `Ctrl+E` | Export results |

---

## Mobile Optimization

The interface is responsive and works on:
- Desktop (recommended for full features)
- Tablet (supported with touch controls)
- Mobile (limited - camera only recommended)

**Mobile-Specific Features**:
- Touch-friendly buttons
- Swipe to navigate images
- Pinch to zoom heatmap
- Portrait/landscape modes

---

## Accessibility

Features for accessibility:
- High contrast mode
- Keyboard navigation
- Screen reader support
- Text sizing options
- Color-blind friendly palette

---

## Tips & Best Practices

### Image Capture Tips
1. **Lighting**: Use consistent, bright lighting
2. **Angle**: Capture perpendicular to fabric surface
3. **Focus**: Ensure sharp focus, not blurry
4. **Size**: Fill 70-90% of frame with fabric
5. **Contrast**: Avoid shadows and overexposure

### Optimal Settings for Different Scenarios

**High-Speed Production Line**
- Mode: Auto Detect
- Quality: Fast
- Support Shots: 3 (if Unknown)

**Detailed Quality Control**
- Mode: Unknown (few-shot)
- Quality: High
- Support Shots: 10
- Manual measurement: Yes

**Mixed Fabric Types**
- Mode: Auto Detect
- Pattern Mode: Unknown
- Support Shots: 5

### Common Pitfalls to Avoid
- ❌ Blurry images
- ❌ Poor lighting
- ❌ Wrinkled fabric
- ❌ Extreme angles
- ❌ Dirty camera lens
- ✅ Clean, well-lit, focused images

---

## Troubleshooting Interface Issues

### Camera Not Working
1. Check browser permissions
2. Try different camera device
3. Restart browser
4. Update GPU drivers

### Slow Processing
1. Reduce image quality
2. Disable heatmap blending
3. Use Fast quality mode
4. Check GPU utilization

### Results Not Displaying
1. Clear browser cache
2. Refresh page (F5)
3. Check file permissions
4. Restart server

---

**Last Updated**: May 2026
