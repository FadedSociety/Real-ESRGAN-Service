# Real-ESRGAN for Suwayomi Server

**AI-powered 4x manga image upscaling service for Suwayomi Server**

This is a specialized Real-ESRGAN service designed to automatically enhance manga images in Suwayomi Server using neural network upscaling.

## 🚀 Features

- **Automatic Integration**: Auto-detected by Suwayomi Server when running
- **4x Upscaling**: Enhances manga image quality using AI neural networks
- **Memory Optimized**: Tile-based processing for handling large images
- **GPU Acceleration**: CUDA support for faster processing
- **Web Configuration**: Easy setup via browser interface at `localhost:8083`
- **Multiple Models**: Support for different Real-ESRGAN model files
- **Graceful Fallback**: Suwayomi continues working normally when service is offline

## 📋 Requirements

- **Python 3.7+**
- **NVIDIA GPU** (recommended) or CPU
- **At least 4GB RAM** (8GB+ recommended for GPU)
- **Suwayomi Server** with Real-ESRGAN integration

## ⚡ Quick Setup

1. **Run the setup script**:
   ```
   Double-click setup.bat
   ```

2. **Choose your hardware**:
   - Press `1` if you have an NVIDIA GPU (faster)
   - Press `2` if you have CPU only (slower but works)

3. **Wait for installation** - This will:
   - Install Python dependencies
   - Download the AI model (~6MB)
   - Test the service

4. **Start the service**:
   ```
   python memory_optimized_real_esrgan_app.py
   ```

5. **Start Suwayomi Server** - It will automatically detect and use the upscaling service!

## 🎛️ Configuration

Visit `http://localhost:8083` in your browser to:
- Select different AI models
- Adjust tile size for memory management
- Change upscale factor (2x, 3x, 4x)
- Monitor service status

## 📁 File Organization

You can organize your models cleanly:
```
Real-ESRGAN-Service/
├── memory_optimized_real_esrgan_app.py
├── setup.bat
├── simple_setup.py
├── models/
│   ├── RealESRGAN_x4plus_anime_6B.pth
│   └── other_models.pth
└── README.md
```

## 🔧 Manual Setup (Advanced)

If the automatic setup doesn't work:

1. **Install dependencies**:
   ```bash
   # For NVIDIA GPU users:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

   # For CPU users:
   pip install torch torchvision

   # Common dependencies:
   pip install pillow opencv-python flask psutil numpy
   ```

2. **Download model** (if not already downloaded):
   ```bash
   wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
   ```

## 🔗 Integration with Suwayomi

This service integrates automatically with Suwayomi Server:

1. **Auto-Detection**: Suwayomi automatically finds the service when both are running
2. **Seamless Processing**: Images are upscaled transparently during manga reading
3. **Fallback**: Normal manga reading continues if service is offline
4. **Configurable**: Advanced users can override settings via Suwayomi's GraphQL interface

## 🐛 Troubleshooting

### Service won't start
- Make sure Python is installed and in your PATH
- Check that all dependencies are installed
- Try running `python simple_setup.py` manually

### GPU not detected
- Install latest NVIDIA drivers
- Reinstall PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### Out of memory errors
- Reduce tile size in the web interface (try 128 instead of 256)
- Close other GPU-intensive applications
- Consider using CPU mode instead

### Suwayomi not detecting service
- Make sure both services are running
- Check that the service is accessible at `localhost:8083/health`
- Wait up to 1 minute for auto-detection to refresh

## 📊 Performance

- **GPU (NVIDIA)**: ~1-3 seconds per manga page
- **CPU**: ~10-30 seconds per manga page
- **Memory**: 2-6GB depending on image size and tile settings

## 🎯 For Suwayomi Users

This is a companion service for **Suwayomi Server** (manga server). It automatically enhances manga image quality without any manual intervention. Just start both services and enjoy higher quality manga reading!

For more information about Suwayomi Server, visit: https://github.com/Suwayomi/Suwayomi-Server

---

**Note**: This service is optimized specifically for manga/anime images. For other types of images, consider using the original Real-ESRGAN project.
