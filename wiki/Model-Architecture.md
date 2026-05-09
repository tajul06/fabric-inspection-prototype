# Model Architecture Overview

## System Architecture

```
┌─────────────────┐
│   User Input    │
│  (Image/Camera) │
└────────┬────────┘
         │
         ▼
┌──────────────────────────┐
│  Image Preprocessing     │
│  - CLAHE Enhancement     │
│  - Resize & Normalize    │
└────────┬─────────────────┘
         │
         ├─────────────────┬──────────────────┐
         ▼                 ▼                  ▼
    ┌─────────┐     ┌────────────┐    ┌──────────┐
    │ Fabric  │     │ Pattern    │    │ Anomaly  │
    │Classifier   │Classification│    │Detection │
    │(ResNet50)   │(EfficientNet)    │(PatchCore)
    └─────────┘     └────────────┘    └──────────┘
         │                 │                  │
         ▼                 ▼                  ▼
    ┌─────────────────────────────────────────┐
    │      Defect Analysis & Localization     │
    │  - Connected Components                 │
    │  - Bounding Box Detection               │
    │  - Severity Scoring                     │
    └──────────┬──────────────────────────────┘
               │
               ▼
    ┌─────────────────────────────┐
    │    Decision & Report        │
    │  - Pass/Hold/Reject         │
    │  - Confidence Scores        │
    │  - Visualizations           │
    └──────────┬──────────────────┘
               │
               ▼
    ┌─────────────────────────────┐
    │     Web Interface Output    │
    │  - Dashboard Display        │
    │  - Database Storage         │
    │  - Result Export            │
    └─────────────────────────────┘
```

## Model Components

### 1. Fabric Type Classifier

**Architecture**: ResNet50 (Residual Networks)

```
Input Image (3×256×256)
      │
      ▼
┌─────────────────────────────────┐
│ Initial Convolution Layer       │ 7×7, 64 filters, stride 2
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Residual Blocks                 │
│ - Block 1: 64 filters (×3)      │
│ - Block 2: 128 filters (×4)     │
│ - Block 3: 256 filters (×6)     │
│ - Block 4: 512 filters (×3)     │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Global Average Pooling          │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Fully Connected Layers          │
│ - Linear (512 → 256)            │
│ - ReLU Activation               │
│ - Linear (256 → 2)              │ 2 classes: Woven/Knitted
└────────┬────────────────────────┘
         │
         ▼
      Output
  (Class Probabilities)
```

**Key Features**:
- Skip connections prevent vanishing gradients
- Efficient learning from medium-sized datasets
- Fast inference (< 50ms)
- Robust to input variations

---

### 2. Pattern Classifier

**Architecture**: EfficientNet

```
Input Image (3×224×224)
      │
      ▼
┌─────────────────────────────────┐
│ Stem: Conv + BatchNorm + MaxPool│
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ MBConv Blocks (Depthwise-Sep)   │
│ - Stage 1: 16 channels (×3)     │
│ - Stage 2: 24 channels (×5)     │
│ - Stage 3: 40 channels (×5)     │
│ - Stage 4: 80 channels (×7)     │
│ - Stage 5: 112 channels (×5)    │
│ - Stage 6: 192 channels (×7)    │
│ - Stage 7: 320 channels (×1)    │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Head Layer                      │
│ - Conv + BatchNorm + GlobalPool │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Classification Head             │
│ - Fully Connected (1280 → N)    │ N = number of patterns
└────────┬────────────────────────┘
         │
         ▼
      Output
  (Pattern Probabilities)
```

**Advantages**:
- Mobile-efficient: fewer parameters
- Strong accuracy with limited training data
- Compound scaling (depth, width, resolution)
- Great for few-shot learning

---

### 3. Anomaly Detection Models

#### PatchCore
```
Input Image
    │
    ▼
┌─────────────────────────────────┐
│ Feature Extraction              │
│ (Pre-trained CNN backbone)      │
│ - ImageNet pre-trained weights  │
│ - Remove classification head    │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Memory Bank (Normal Examples)    │
│ - Store embeddings from N-class  │
│ - Coreset subsampling           │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Nearest Neighbor Search         │
│ - KNN in feature space          │
│ - Compute anomaly distance      │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Anomaly Score Generation        │
│ - Per-patch scores              │
│ - Upsampling to image size      │
│ - Heatmap generation            │
└────────┬────────────────────────┘
         │
         ▼
    Output
 (Anomaly Map + Score)
```

**Key Properties**:
- No training required on normal examples
- Works with as few as 1-5 normal images
- Training-free adaptation
- Fast inference

#### WinCLIP (Few-Shot Mode)
```
Normal Reference Images (Support Set)
          │
          ▼
┌──────────────────────────────┐
│ CLIP Feature Extraction      │
│ - Vision Transformer         │
│ - Pre-trained on 400M pairs  │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│ Prototype Generation         │
│ - Average features           │
│ - Normalize                  │
└────────┬─────────────────────┘
         │
         ▼
   Query Image
         │
         ▼
┌──────────────────────────────┐
│ CLIP Feature Extraction      │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│ Similarity Computation       │
│ - Cosine similarity          │
│ - Anomaly = 1 - similarity   │
└────────┬─────────────────────┘
         │
         ▼
    Output
 (Anomaly Score)
```

---

### 4. Defect Analysis Pipeline

```
Anomaly Heatmap
     │
     ▼
┌──────────────────────────────────┐
│ Thresholding                     │
│ - Convert to binary mask         │
│ - Confidence threshold           │
└─────────┬──────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│ Connected Components Labeling    │
│ - Find connected regions         │
│ - Label each defect             │
└─────────┬──────────────────────┘
          │
          ├──────────────┬──────────────┐
          ▼              ▼              ▼
     ┌─────────┐  ┌──────────┐  ┌─────────┐
     │ Bounding│  │   Area   │  │Severity │
     │  Boxes  │  │Calculation  │ Scoring │
     └─────────┘  └──────────┘  └─────────┘
          │
          └──────────────┬──────────────┘
                         │
                         ▼
┌──────────────────────────────────┐
│ Final Decision                   │
│ - Pass (score < 0.3)             │
│ - Hold (0.3 ≤ score < 0.7)       │
│ - Reject (score ≥ 0.7)           │
└──────────────────────────────────┘
```

---

## Data Flow in Training

### Transfer Learning Approach

```
ImageNet Pre-trained Models
     │
     ├─────────────────────┬─────────────────────┐
     ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  ResNet50    │    │ EfficientNet │    │ CLIP Vision  │
│  Backbone    │    │  Backbone    │    │ Transformer  │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       │                   ▼                   │
       │            ┌──────────────────┐      │
       │            │ Pattern Classifier   │      │
       │            │ Fine-tuning Dataset │      │
       │            └──────────────────┘      │
       │                                      │
       ▼                                      ▼
┌──────────────────┐              ┌──────────────────┐
│ Fabric Classifier│              │ Anomaly Detector │
│ Fine-tuning      │              │ (PatchCore prep) │
└──────────────────┘              └──────────────────┘
```

---

## Model Parameters Summary

| Model | Backbone | Parameters | Input Size | Output |
|-------|----------|-----------|------------|--------|
| Fabric Classifier | ResNet50 | 25.5M | 256×256×3 | 2 classes |
| Pattern Classifier (Woven) | EfficientNet-B3 | 10.8M | 224×224×3 | N patterns |
| Pattern Classifier (Knitted) | EfficientNet-B3 | 10.8M | 224×224×3 | N patterns |
| PatchCore | ResNet18/50 | 11.2M/25.5M | Variable | Anomaly map |
| WinCLIP | ViT-B/32 | 88M | 224×224×3 | Similarity score |

---

## Performance Characteristics

### Inference Speed (GPU: RTX 3090)
- Fabric Classification: ~50ms
- Pattern Classification: ~100ms
- Anomaly Detection: ~200ms
- Total (including I/O): ~400-500ms per image

### Memory Usage
- Model loading: ~2GB
- Batch inference (1 image): ~500MB
- Peak during all models loaded: ~3GB

### Accuracy Benchmarks
- Fabric Classification: 94-97%
- Pattern Classification: 88-93%
- Anomaly Detection (F1-score): 85-92%

---

## Extension Points

### Adding New Pattern Classifiers
1. Train EfficientNet on your pattern categories
2. Add to `Fabric_pattern_classifier/` directory
3. Update `ml_models.py` to load new model
4. Register in configuration

### Custom Anomaly Detectors
1. Train PatchCore on your fabric/defect combinations
2. Save to `models/` directory
3. Update model discovery logic
4. Reference in routes

---

**Last Updated**: May 2026
