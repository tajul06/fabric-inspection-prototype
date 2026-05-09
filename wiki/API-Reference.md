# API Reference

Complete HTTP API documentation for the Fabric Inspection system.

## Base URL

```
http://localhost:5000
http://<your-ip>:5000  (for remote access)
```

## Authentication

Currently, the API does not require authentication. For production deployment, implement token-based authentication.

## Endpoints

### Dashboard & Views

#### GET /
**Description**: Main dashboard page  
**Response**: HTML dashboard view  
**Status Codes**:
- `200` - Success

**Example**:
```bash
curl http://localhost:5000/
```

---

#### GET /inspection
**Description**: Inspection interface page  
**Response**: HTML inspection form  
**Status Codes**:
- `200` - Success

**Example**:
```bash
curl http://localhost:5000/inspection
```

---

### Image Processing

#### POST /inspection
**Description**: Submit fabric image for inspection and analysis  
**Content-Type**: `multipart/form-data`

**Request Parameters**:
```json
{
  "image": "file",                    // Required: JPG/PNG image
  "fabric_type": "string",            // Optional: "woven" or "knitted"
  "pattern_mode": "string",           // Optional: "auto" or "unknown"
  "support_shots": "integer",         // Optional: 3, 5, or 10 (for unknown mode)
  "measurement_mode": "string",       // Optional: "auto_ratio" or "manual"
  "width_cm": "float",                // Optional: fabric width in cm
  "height_cm": "float"                // Optional: fabric height in cm
}
```

**Response**:
```json
{
  "success": true,
  "result_id": "uuid",
  "fabric_type": "woven",
  "pattern": "stripe",
  "anomaly_score": 0.42,
  "decision": "PASS",
  "defect_count": 2,
  "defects": [
    {
      "id": 1,
      "area_cm2": 15.5,
      "severity": "medium",
      "location": "top-right"
    }
  ],
  "image_paths": {
    "original": "/static/original_xyz.jpg",
    "preprocessed": "/static/preprocessed_xyz.jpg",
    "heatmap": "/static/heatmap_xyz.jpg",
    "annotated": "/static/annotated_xyz.jpg"
  },
  "confidence": 0.91,
  "processing_time_ms": 450
}
```

**Status Codes**:
- `200` - Success
- `400` - Invalid parameters
- `413` - File too large (> 25MB)
- `500` - Server error

**Example**:
```bash
curl -X POST http://localhost:5000/inspection \
  -F "image=@fabric.jpg" \
  -F "fabric_type=woven" \
  -F "pattern_mode=auto"
```

---

### Heatmap Blending

#### POST /blend_api
**Description**: Generate blended heatmap with custom transparency  
**Content-Type**: `application/json`

**Request**:
```json
{
  "original_image": "path/to/original.jpg",
  "heatmap_image": "path/to/heatmap.jpg",
  "alpha": 0.5                    // Transparency: 0.0-1.0
}
```

**Response**:
```json
{
  "success": true,
  "blended_image": "data:image/png;base64,...",
  "alpha_used": 0.5
}
```

**Status Codes**:
- `200` - Success
- `400` - Missing parameters
- `404` - Image not found
- `500` - Processing error

**Example**:
```bash
curl -X POST http://localhost:5000/blend_api \
  -H "Content-Type: application/json" \
  -d '{
    "original_image": "path/to/original.jpg",
    "heatmap_image": "path/to/heatmap.jpg",
    "alpha": 0.6
  }'
```

---

### Local Anomaly Score

#### POST /local_score_api
**Description**: Get anomaly score at specific pixel location  
**Content-Type**: `application/json`

**Request**:
```json
{
  "heatmap_image": "path/to/heatmap.jpg",
  "x": 150,                       // Pixel X coordinate
  "y": 100                        // Pixel Y coordinate
}
```

**Response**:
```json
{
  "success": true,
  "local_score": 0.75,
  "severity": "high",
  "x": 150,
  "y": 100,
  "suggested_type": "hole"
}
```

**Status Codes**:
- `200` - Success
- `400` - Invalid coordinates
- `404` - Image not found
- `500` - Processing error

**Example**:
```bash
curl -X POST http://localhost:5000/local_score_api \
  -H "Content-Type: application/json" \
  -d '{
    "heatmap_image": "processed/heatmap_xyz.jpg",
    "x": 200,
    "y": 150
  }'
```

---

### Pipeline Preview

#### POST /pipeline_preview_api
**Description**: Preview image processing pipeline (CLAHE, resizing, etc.)  
**Content-Type**: `multipart/form-data`

**Request**:
```json
{
  "image": "file"                 // Image to process
}
```

**Response**:
```json
{
  "success": true,
  "original": "data:image/png;base64,...",
  "clahe": "data:image/png;base64,...",
  "resized": "data:image/png;base64,...",
  "normalized": "data:image/png;base64,..."
}
```

**Status Codes**:
- `200` - Success
- `400` - Invalid image
- `500` - Processing error

**Example**:
```bash
curl -X POST http://localhost:5000/pipeline_preview_api \
  -F "image=@fabric.jpg"
```

---

### File Download

#### GET /result_file/<filename>
**Description**: Download processed result image  
**Response**: Binary image file  

**Parameters**:
- `filename` - Result image filename (e.g., `heatmap_xyz.jpg`)

**Status Codes**:
- `200` - Success
- `404` - File not found
- `403` - Forbidden (path traversal protection)

**Example**:
```bash
curl -O http://localhost:5000/result_file/heatmap_abc123.jpg
```

---

## Response Codes

| Code | Meaning |
|------|---------|
| 200 | Successful request |
| 400 | Bad request (missing/invalid parameters) |
| 404 | Resource not found |
| 413 | Payload too large |
| 500 | Internal server error |

---

## Error Handling

### Error Response Format

```json
{
  "success": false,
  "error": "Error message describing what went wrong",
  "code": "ERROR_CODE"
}
```

### Common Errors

**Invalid Image Format**
```json
{
  "success": false,
  "error": "Unsupported image format. Use JPG or PNG.",
  "code": "INVALID_FORMAT"
}
```

**Model Not Found**
```json
{
  "success": false,
  "error": "Required model not found. Check model configuration.",
  "code": "MODEL_NOT_FOUND"
}
```

**CUDA Out of Memory**
```json
{
  "success": false,
  "error": "GPU out of memory. Try smaller image or restart server.",
  "code": "CUDA_OOM"
}
```

---

## Rate Limiting

Currently, no rate limiting is implemented. For production, add:

```python
from flask_limiter import Limiter
limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/inspection', methods=['POST'])
@limiter.limit("5 per minute")
def inspect():
    ...
```

---

## Example Workflows

### Complete Inspection Workflow

```bash
# 1. Upload and inspect image
RESPONSE=$(curl -X POST http://localhost:5000/inspection \
  -F "image=@fabric.jpg" \
  -F "fabric_type=woven")

# 2. Extract result ID
RESULT_ID=$(echo $RESPONSE | jq '.result_id')

# 3. Get heatmap path
HEATMAP=$(echo $RESPONSE | jq '.image_paths.heatmap')

# 4. Blend heatmap with custom alpha
curl -X POST http://localhost:5000/blend_api \
  -H "Content-Type: application/json" \
  -d "{
    \"original_image\": \"$(echo $RESPONSE | jq '.image_paths.original')\",
    \"heatmap_image\": \"$HEATMAP\",
    \"alpha\": 0.7
  }"
```

### Python Integration

```python
import requests
import json

# Inspect image
files = {'image': open('fabric.jpg', 'rb')}
data = {'fabric_type': 'woven', 'pattern_mode': 'auto'}
response = requests.post('http://localhost:5000/inspection', 
                        files=files, data=data)

result = response.json()
print(f"Decision: {result['decision']}")
print(f"Anomaly Score: {result['anomaly_score']}")
print(f"Defects Found: {result['defect_count']}")

# Get local score
local_data = {
    'heatmap_image': result['image_paths']['heatmap'],
    'x': 150,
    'y': 100
}
local_response = requests.post('http://localhost:5000/local_score_api',
                              json=local_data)
local_result = local_response.json()
print(f"Local Anomaly Score: {local_result['local_score']}")
```

---

## CORS Headers

For cross-origin requests, consider adding:

```python
from flask_cors import CORS
CORS(app)
```

---

## API Versioning

Future API versions will follow: `/api/v1/`, `/api/v2/`, etc.

Current version: **v0** (unstable, breaking changes possible)

---

**Last Updated**: May 2026
