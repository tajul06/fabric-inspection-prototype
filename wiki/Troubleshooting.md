# Troubleshooting Guide

## Common Issues and Solutions

### Camera Issues

#### "Camera access not supported"
**Cause**: Browser security restrictions  
**Solutions**:
1. Use HTTPS in production (localhost works without it for testing)
2. Check browser permissions (Chrome → Settings → Privacy → Camera)
3. Ensure camera is not in use by other applications
4. Try a different browser (Chrome, Firefox, Edge)

#### "Camera preview shows garbled video"
**Cause**: Codec incompatibility  
**Solutions**:
1. Try a different camera device from the dropdown
2. Disable browser hardware acceleration
3. Update GPU drivers
4. For DroidCam: Switch to MJPEG format in app settings

#### DroidCam not appearing in device list
**Cause**: Network/connection issue  
**Solutions**:
1. Verify phone and laptop are on the same Wi-Fi
2. Note the IP:Port shown in DroidCam app
3. Restart DroidCam on phone
4. Allow firewall exceptions
5. Test connection with ping

```bash
ping <phone-ip>
```

---

### Model Loading Issues

#### "ModuleNotFoundError: No module named 'anomalib'"
**Cause**: Missing dependency  
**Solution**:
```bash
pip install anomalib[full]
# or specific version
pip install "anomalib==1.0.1"
```

#### "RuntimeError: CUDA out of memory"
**Cause**: GPU memory exhausted  
**Solutions**:
1. Reduce input image size in `app/config.py`
2. Process one image at a time (default behavior)
3. Close other GPU-intensive applications
4. Use CPU mode (`CUDA_VISIBLE_DEVICES=""`)

#### "No PatchCore model found"
**Cause**: Model file missing or incorrect path  
**Solutions**:
1. Check paths in `app/config.py`
2. Verify model files exist in specified directories
3. Copy models to portable `models/` folder
4. Check file permissions

---

### Inference Issues

#### "Operands could not be broadcast together"
**Cause**: Image dimension mismatch  
**Solutions**:
1. Check if image is valid (not corrupted)
2. Ensure image format is supported (JPG, PNG)
3. Try a different image
4. Restart the Flask server

#### "Inference is very slow"
**Cause**: Running on CPU or large image size  
**Solutions**:

**For CPU users**:
1. Install CUDA if you have compatible GPU
2. Use smaller images
3. Reduce model complexity if possible

**For GPU users**:
```bash
# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU utilization
nvidia-smi  # On Windows: check Task Manager → Performance → GPU
```

#### "Wrong or inconsistent predictions"
**Cause**: Model not trained on your fabric type  
**Solutions**:
1. Use Unknown pattern mode with WinCLIP
2. Provide more diverse support images
3. Check if fabric type is correctly detected
4. Train custom model on your fabric type

---

### Web Interface Issues

#### "Image doesn't upload"
**Cause**: File size or format issue  
**Solutions**:
1. Check file size (default max 25MB)
2. Use common formats: JPG, PNG
3. Verify file is not corrupted
4. Clear browser cache
5. Try different browser

#### "Page takes too long to load"
**Cause**: Large result files or slow server  
**Solutions**:
1. Clear processed/ directory to free space
2. Restart Flask server
3. Check network connection speed
4. Use a faster computer or increase resources

#### "Heatmap blending not working"
**Cause**: Browser canvas rendering issue  
**Solutions**:
1. Update browser to latest version
2. Disable browser extensions
3. Clear cache and cookies
4. Try in Incognito/Private mode

---

### Configuration Issues

#### "SECRET_KEY warning" in logs
**Cause**: Using development secret key  
**Solution**: Update `app/config.py` for production
```python
import secrets
SECRET_KEY = secrets.token_hex(32)
```

#### "Database locked" error
**Cause**: Multiple processes accessing SQLite simultaneously  
**Solutions**:
1. Ensure only one Flask instance is running
2. Close other database connections
3. Delete `*.db-journal` file if it exists
4. Use production database (PostgreSQL) for multi-user

#### "Upload folder not found"
**Cause**: Missing directory  
**Solution**:
```bash
mkdir uploads
mkdir processed
```

---

### Performance Issues

#### Low accuracy on defect detection
**Cause**: Model not trained on your fabric type  
**Solutions**:
1. Use correct fabric type in settings
2. Use appropriate pattern mode
3. Check image quality (focus, lighting)
4. Train custom model on your fabrics

#### High false positive rate
**Cause**: Anomaly threshold too low  
**Solutions**:
1. Increase anomaly threshold in results
2. Provide better quality reference images
3. Use more support images (10 instead of 5)
4. Verify fabric type classification

#### Model training is slow
**Cause**: CPU training or large dataset  
**Solutions**:
1. Use GPU for training
2. Use smaller batches
3. Reduce image resolution
4. Use pre-trained models (transfer learning)

---

### Windows-Specific Issues

#### ".bat file doesn't execute"
**Cause**: Execution policy or path issues  
**Solutions**:
```powershell
# Allow script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or run directly with PowerShell
powershell -ExecutionPolicy Bypass -File build_cpu.bat
```

#### "Python not found in build script"
**Cause**: Python not in PATH  
**Solutions**:
1. Use full path to Python: `C:\Python310\python.exe`
2. Add Python to Windows PATH
3. Use virtual environment activate script first

#### Port 5000 already in use
**Cause**: Another application using the port  
**Solutions**:
```powershell
# Find process using port 5000
netstat -ano | findstr :5000

# Kill process (replace PID with actual number)
taskkill /PID <PID> /F

# Or use different port in app/config.py
PORT = 5001
```

---

### Network/Deployment Issues

#### "Connection refused" from remote device
**Cause**: Firewall blocking or wrong IP  
**Solutions**:
1. Find your LAN IP: `ipconfig`
2. Allow port 5000 in Windows Firewall
3. Verify both devices on same network
4. Disable VPN if interfering

#### HTTPS/SSL errors
**Cause**: Self-signed certificate or missing certificate  
**Solutions**:
1. For production, get valid SSL certificate
2. For development, use HTTP on localhost
3. Accept browser certificate warning if needed

---

### Getting Help

If you can't find a solution:

1. Check existing GitHub issues
2. Search the wiki thoroughly
3. Create a new issue with:
   - Error message (full traceback)
   - Steps to reproduce
   - Environment info (OS, Python, CUDA version)
   - Sample image if possible
4. Include logs from terminal/console

---

**Last Updated**: May 2026
