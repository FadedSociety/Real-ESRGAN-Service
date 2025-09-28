# Real-ESRGAN HTTP Service

A memory-optimized Real-ESRGAN upscaling service with web interface and HTTP API for 4x image enhancement.

## Features

- **GPU Accelerated**: CUDA support with automatic GPU detection
- **Memory Optimized**: Configurable tile processing for memory efficiency
- **Web Interface**: Configuration panel at `http://localhost:8083`
- **Dynamic Configuration**: Real-time model switching and tile size adjustment
- **HTTP API**: RESTful endpoints for image upscaling
- **Model Selection**: Automatic .pth file detection and switching
- **Health Monitoring**: Memory usage tracking and service status

## Quick Start

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- PyTorch with CUDA

### Installation

1. **Install dependencies**:
   ```bash
   python setup.py
   ```

2. **Start the service**:
   ```bash
   python memory_optimized_real_esrgan_app.py
   ```

3. **Access web interface**:
   Open `http://localhost:8083` in your browser

## API Endpoints

### Health Check
```http
GET /health
```
Returns service status and current configuration.

### Configuration
```http
POST /config
Content-Type: application/x-www-form-urlencoded

model_path=RealESRGAN_x4plus_anime_6B.pth&scale_factor=4&tile_size=512
```

### Image Upscaling
```http
POST /upscale
Content-Type: multipart/form-data

image: [binary image data]
```

## Configuration Options

- **Model Path**: Select from available .pth model files
- **Scale Factor**: 2x, 3x, or 4x upscaling (model dependent)
- **Tile Size**: 128, 256, or 512 pixels (affects memory usage)

## Integration

This service is designed to work with:
- [Suwayomi-Server](https://github.com/FadedSociety/Suwayomi-Server) HTTP converter
- Any application requiring HTTP-based image upscaling
- Standalone web interface for manual upscaling

## Files

- `memory_optimized_real_esrgan_app.py` - Main Flask service
- `rrdb_arch.py` - Neural network architecture
- `setup.py` - Dependency installer
- `.gitignore` - Git ignore patterns

## Hardware Requirements

- **Minimum**: 4GB VRAM, 8GB RAM
- **Recommended**: 8GB+ VRAM, 16GB+ RAM
- **Tile Size Guide**:
  - 128px: Low memory usage, slower processing
  - 256px: Balanced performance
  - 512px: High memory usage, faster processing

## License

This project builds upon [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by Xintao Wang et al.

---

Generated with [Claude Code](https://claude.ai/code)