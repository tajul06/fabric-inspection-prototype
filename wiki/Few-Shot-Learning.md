# Few-Shot Learning Explained

## What is Few-Shot Learning?

Few-shot learning is a machine learning paradigm where a model learns to recognize new concepts from a very limited number of examples (typically 1-10 images per class).

## Traditional vs Few-Shot Learning

### Traditional Supervised Learning
- Requires hundreds or thousands of labeled examples per class
- Trains from scratch for each new task
- High annotation cost
- Time-consuming process

### Few-Shot Learning
- Learns from 3-10 examples per class
- Quickly adapts to new tasks
- Low annotation cost
- Rapid deployment

## How It Works in This Project

### 1. Prototypical Networks

The framework uses prototypical networks for few-shot classification:

1. **Support Set**: A small set of reference images (e.g., 5 normal fabric images)
2. **Query Set**: Test images to classify
3. **Feature Extraction**: Both support and query images are embedded into a metric space
4. **Classification**: Query images are classified based on distance to class prototypes

### 2. WinCLIP for Anomaly Detection

When you use "Unknown" pattern mode:

1. You provide 3/5/10 reference images of normal (good) fabric
2. The system learns what "normal" looks like from these references
3. When inspecting new images, anomalies are detected as deviations from the reference set

## Applications in Fabric Inspection

### Adapting to New Fabrics
```
Scenario: You have a new fabric type not in the training data

Traditional Approach:
- Collect 500+ images of the new fabric
- Annotate defects in all images
- Retrain model (hours of computation)
- Deploy new model

Few-Shot Approach:
- Capture 5-10 images of normal fabric
- Load into Unknown pattern mode
- Immediately start inspection
- System detects anomalies relative to your reference set
```

### Benefits
1. **Speed**: Deploy in minutes, not weeks
2. **Cost**: Minimal annotation effort
3. **Flexibility**: Adapt to any fabric variant
4. **Learning**: System improves with more examples

## Practical Usage Guide

### Step 1: Prepare Reference Images
```
Collect 5-10 images of normal (defect-free) fabric:
- Different lighting conditions
- Different angles
- Typical variations
```

### Step 2: Configure in Web UI
```
1. Set Pattern Mode → "Unknown"
2. Set Support Shots → 5 (or 3/10)
3. Click "Capture Support Images"
4. Upload your reference images
```

### Step 3: Inspect Fabric
```
1. Upload inspection image
2. Click "Inspect"
3. System compares against your reference set
4. Anomaly score shows deviation from references
```

### Step 4: Refine (Optional)
```
If results are off:
- Add more support images (10 instead of 5)
- Use images with better lighting
- Include edge cases
- Re-run inspection
```

## Model Architectures Used

### CLIP (Contrastive Language-Image Pre-training)
- Pre-trained on 400M image-text pairs
- Understands both visual and semantic information
- Strong few-shot learning capability
- Backbone for WinCLIP

### Prototypical Networks
- Simple yet effective
- Metric learning approach
- Fast inference
- Scalable to many classes

### PatchCore
- State-of-the-art anomaly detection
- Uses pre-trained feature extractors
- Works with limited normal examples
- Strong generalization

## Key Concepts

### Metric Space Learning
The models learn to embed images into a space where:
- Similar images are close together
- Anomalies are far from normal examples
- Classification is based on distance

### Support Set Quality
The quality of reference images directly affects performance:
- **Good**: Diverse, well-lit, representative
- **Bad**: Single lighting, single angle, narrow distribution

### Transfer Learning
The models use pre-trained backbones (ImageNet, CLIP) which provide:
- Rich feature representations
- Domain knowledge
- Strong generalization

## Limitations & Considerations

1. **Support Set Bias**: If your reference set has biases, the model will too
2. **Domain Shift**: Large differences from training data require more examples
3. **Lighting**: Very different lighting than references may reduce accuracy
4. **Pattern Complexity**: Complex patterns need more diverse references

## Best Practices

1. **Diverse References**: Use different angles, lighting, positions
2. **Representative**: Include typical variations in your fabric
3. **Quality**: Use clear, well-focused images
4. **Quantity**: 5-10 examples usually sufficient; 3 is minimum

## Further Reading

- Snell, J., Swersky, K., & Zemel, R. (2017). "Prototypical Networks for Few-shot Learning"
- Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision"
- Roth, K., et al. (2022). "Towards Total Recall in Industrial Anomaly Detection with Prototype-Guided Mask"

---

**Last Updated**: May 2026
