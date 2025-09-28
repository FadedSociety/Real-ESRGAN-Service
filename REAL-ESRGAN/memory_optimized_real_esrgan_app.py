#!/usr/bin/env python3
"""
Memory-Optimized Real-ESRGAN Upscaling Service
Auto-checks dependencies and provides helpful setup guidance
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check and provide guidance for missing dependencies"""
    missing_deps = []

    # Check required packages
    required_packages = {
        'torch': 'torch',
        'PIL': 'pillow',
        'cv2': 'opencv-python',
        'flask': 'flask',
        'psutil': 'psutil',
        'numpy': 'numpy'
    }

    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_deps.append(package_name)

    if missing_deps:
        print("[ERROR] Missing dependencies detected!")
        print(f"   Missing: {', '.join(missing_deps)}")
        print()
        print("[FIX] Quick Fix:")
        print("   python setup.py")
        print()
        print("[MANUAL] Manual Install:")
        print(f"   pip install {' '.join(missing_deps)}")
        print()
        print("[GPU] For GPU acceleration, you also need:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        sys.exit(1)

    # Check model file
    from pathlib import Path
    model_path = Path("RealESRGAN_x4plus_anime_6B.pth")
    if not model_path.exists():
        print("[ERROR] Model file missing!")
        print("   Expected: RealESRGAN_x4plus_anime_6B.pth")
        print()
        print("[FIX] Quick Fix:")
        print("   python setup.py")
        print()
        print("[MANUAL] Manual Download:")
        print("   wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth")
        sys.exit(1)

    # Check CUDA availability
    import torch
    if not torch.cuda.is_available():
        print("[WARNING] CUDA not detected - will use CPU (much slower)")
        print("[GPU] For GPU acceleration:")
        print("   1. Install NVIDIA drivers")
        print("   2. Run: python setup.py")
        print()

# Run dependency check
print("[CHECK] Checking dependencies...")
check_dependencies()
print("[SUCCESS] All dependencies found!")

import gc
import psutil
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_file
import numpy as np
from PIL import Image
import io
from rrdb_arch import RRDBNet
import threading
import time

app = Flask(__name__)

# Memory monitoring
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# Global variables - initialize once
device = None
model = None
current_model_path = None
config_tile_size = 256
config_scale_factor = 4
model_lock = threading.Lock()
last_cleanup = time.time()

def force_memory_cleanup():
    """Aggressive memory cleanup"""
    global last_cleanup
    current_time = time.time()

    # Only cleanup every 30 seconds to avoid overhead
    if current_time - last_cleanup < 30:
        return

    print(f"Memory before cleanup: {get_memory_usage():.1f} MB")

    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Force garbage collection
    for _ in range(3):
        gc.collect()

    last_cleanup = current_time
    print(f"Memory after cleanup: {get_memory_usage():.1f} MB")

def initialize_model():
    """Initialize model once and keep it loaded"""
    global device, model, current_model_path

    if model is not None:
        return True

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        if device.type == 'cuda':
            gpu_props = torch.cuda.get_device_properties(0)
            print(f"GPU: {gpu_props.name}")
            print(f"GPU Memory: {gpu_props.total_memory / 1024**3:.1f} GB")

            # Set memory fraction to prevent overallocation
            torch.cuda.set_per_process_memory_fraction(0.7)  # Use max 70% of GPU memory

        # Load model - prioritize models/ folder
        models_dir_path = os.path.join(os.path.dirname(__file__), "models", "RealESRGAN_x4plus_anime_6B.pth")
        if os.path.exists(models_dir_path):
            model_path = models_dir_path
        else:
            # Fallback to current directory
            model_path = "RealESRGAN_x4plus_anime_6B.pth"

        current_model_path = os.path.abspath(model_path)  # Set the current model path
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)

        # Load with memory mapping to reduce RAM usage
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        if 'params_ema' in checkpoint:
            model.load_state_dict(checkpoint['params_ema'], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)

        # Clear checkpoint from memory immediately
        del checkpoint

        model.eval()
        model.to(device)

        # Disable compilation to avoid Triton dependency
        # if hasattr(torch, 'compile') and device.type == 'cuda':
        #     model = torch.compile(model, mode="reduce-overhead")

        print("[SUCCESS] Memory-optimized Real-ESRGAN model loaded successfully!")
        force_memory_cleanup()
        return True

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False

def process_image_tiled(image_tensor, tile_size=256, tile_pad=16):
    """Process image in tiles to reduce memory usage"""
    with model_lock:
        if model is None:
            if not initialize_model():
                return None

        batch, channel, height, width = image_tensor.shape
        scale = config_scale_factor  # Use configurable scale factor

        # If image is small enough, process directly
        if height <= tile_size and width <= tile_size:
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    output = model(image_tensor)
            return output

        # Process in tiles for large images
        tiles_x = (width + tile_size - 1) // tile_size
        tiles_y = (height + tile_size - 1) // tile_size

        output = torch.zeros((batch, channel, height * scale, width * scale), dtype=torch.float32, device=device)

        for y in range(tiles_y):
            for x in range(tiles_x):
                # Calculate tile boundaries
                start_x = x * tile_size
                end_x = min(start_x + tile_size, width)
                start_y = y * tile_size
                end_y = min(start_y + tile_size, height)

                # Extract tile with padding
                tile_start_x = max(start_x - tile_pad, 0)
                tile_end_x = min(end_x + tile_pad, width)
                tile_start_y = max(start_y - tile_pad, 0)
                tile_end_y = min(end_y + tile_pad, height)

                tile = image_tensor[:, :, tile_start_y:tile_end_y, tile_start_x:tile_end_x]

                # Process tile
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        tile_output = model(tile)

                # Calculate output positions using configurable scale
                scale = config_scale_factor
                output_start_x = start_x * scale
                output_end_x = end_x * scale
                output_start_y = start_y * scale
                output_end_y = end_y * scale

                # Calculate crop positions in processed tile
                crop_start_x = (start_x - tile_start_x) * scale
                crop_end_x = crop_start_x + (end_x - start_x) * scale
                crop_start_y = (start_y - tile_start_y) * scale
                crop_end_y = crop_start_y + (end_y - start_y) * scale

                # Place processed tile in output
                output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = \
                    tile_output[:, :, crop_start_y:crop_end_y, crop_start_x:crop_end_x]

                # Clean up tile tensors immediately
                del tile, tile_output
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        return output

@app.route('/')
def index():
    global model, current_model_path
    available_models = []

    # Scan for available models, prioritizing models/ folder
    # First, check models/ folder if it exists
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.pth'):
                full_path = os.path.join(models_dir, file)
                available_models.append(full_path)

    # Then scan current directory and subdirectories for any remaining models
    for root, dirs, files in os.walk('.'):
        # Skip the models directory since we already processed it
        if 'models' in root:
            continue
        for file in files:
            if file.endswith('.pth'):
                full_path = os.path.join(root, file)
                # Avoid duplicates
                if full_path not in available_models:
                    available_models.append(full_path)

    memory_mb = get_memory_usage()

    return f'''<!DOCTYPE html>
<html>
<head>
    <title>Real-ESRGAN Configuration Panel</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; text-align: center; margin-bottom: 30px; }}
        .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
        .endpoint {{ background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 3px; font-family: monospace; }}
        .status {{ color: #28a745; font-weight: bold; }}
        .info {{ color: #666; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f8f9fa; }}
        .form-group {{ margin: 15px 0; }}
        label {{ display: block; margin-bottom: 5px; font-weight: bold; }}
        select, input {{ width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }}
        button {{ background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }}
        button:hover {{ background: #0056b3; }}
        .success {{ color: #28a745; font-weight: bold; }}
        .error {{ color: #dc3545; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Real-ESRGAN Configuration Panel</h1>

        <div class="section">
            <h3>Service Status</h3>
            <p class="status">Service Running - Memory: {memory_mb:.1f} MB</p>
            <p class="info">Configure settings for Suwayomi image upscaling requests</p>
        </div>

        <div class="section">
            <h3>Model Configuration</h3>
            <form action="/config" method="post">
                <div class="form-group">
                    <label>Model File (.pth):</label>
                    <select name="model_path">
                        {"".join([f'<option value="{model}" {"selected" if model == current_model_path else ""}>{model}</option>' for model in available_models])}
                    </select>
                </div>

                <div class="form-group">
                    <label>Upscale Factor:</label>
                    <select name="scale_factor">
                        <option value="2">2x Upscaling</option>
                        <option value="3">3x Upscaling</option>
                        <option value="4" selected>4x Upscaling</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Tile Size (for memory management):</label>
                    <select name="tile_size">
                        <option value="128">128 (Low memory)</option>
                        <option value="256" selected>256 (Balanced)</option>
                        <option value="512">512 (High quality)</option>
                    </select>
                </div>

                <button type="submit">Apply Configuration</button>
            </form>
        </div>

        <div class="section">
            <h3>Current Configuration</h3>
            <table>
                <tr><th>Setting</th><th>Value</th></tr>
                <tr><td>Model</td><td>{current_model_path or "Not loaded"}</td></tr>
                <tr><td>Device</td><td>{"CUDA (GPU)" if torch.cuda.is_available() else "CPU"}</td></tr>
                <tr><td>Model Loaded</td><td>{"Yes" if model else "No"}</td></tr>
                <tr><td>API Port</td><td>8083</td></tr>
            </table>
        </div>

        <div class="section">
            <h3>API Endpoints</h3>
            <div class="endpoint">GET /health - Health check</div>
            <div class="endpoint">POST /upscale - Main upscaling endpoint</div>
            <div class="endpoint">POST /real-esrgan - Alternative endpoint</div>
            <div class="endpoint">POST /anime-upscale - Community-specific endpoint</div>
        </div>

        <div class="section">
            <h3>Suwayomi Integration</h3>
            <p class="info">Configure Suwayomi to use: <code>http://localhost:8083/upscale</code></p>
            <p class="info">The service automatically handles image upscaling requests from Suwayomi</p>
        </div>
    </div>
</body>
</html>'''

@app.route('/config', methods=['POST'])
def update_config():
    global model, current_model_path, config_tile_size, config_scale_factor

    try:
        model_path = request.form.get('model_path')
        scale_factor = int(request.form.get('scale_factor', 4))
        tile_size = int(request.form.get('tile_size', 256))

        # Update configuration
        config_tile_size = tile_size
        config_scale_factor = scale_factor

        # Load new model if different
        if model_path and model_path != current_model_path:
            if os.path.exists(model_path):
                print(f"[CONFIG] Loading new model: {model_path}")

                # Clear old model
                if model:
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Load new model using our existing architecture
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)

                # Load with memory mapping
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                if 'params_ema' in checkpoint:
                    model.load_state_dict(checkpoint['params_ema'], strict=True)
                else:
                    model.load_state_dict(checkpoint, strict=True)

                del checkpoint
                model.eval()
                model.to(device)
                current_model_path = os.path.abspath(model_path)

                print(f"[CONFIG] Model loaded successfully: {model_path}")
                return f'''<html><body><h2>Configuration Updated!</h2>
                <p>Model: {model_path}</p>
                <p>Scale Factor: {scale_factor}x</p>
                <p>Tile Size: {tile_size}</p>
                <p><a href="/">Back to Configuration Panel</a></p></body></html>'''
            else:
                return f'''<html><body><h2>Error!</h2>
                <p>Model file not found: {model_path}</p>
                <p><a href="/">Back to Configuration Panel</a></p></body></html>'''
        else:
            return f'''<html><body><h2>Configuration Updated!</h2>
            <p>Scale Factor: {scale_factor}x</p>
            <p>Tile Size: {tile_size}</p>
            <p><a href="/">Back to Configuration Panel</a></p></body></html>'''

    except Exception as e:
        return f'''<html><body><h2>Error!</h2>
        <p>Failed to update configuration: {str(e)}</p>
        <p><a href="/">Back to Configuration Panel</a></p></body></html>'''

@app.route('/health')
def health_check():
    memory_mb = get_memory_usage()
    return jsonify({
        'status': 'healthy',
        'memory_usage_mb': round(memory_mb, 1),
        'model_loaded': model is not None,
        'current_model': current_model_path or 'Not loaded',
        'device': str(device) if device else 'not_initialized',
        'config': {
            'tile_size': config_tile_size,
            'scale_factor': config_scale_factor,
            'quality': 95
        }
    })

@app.route('/upscale', methods=['POST'])
def upscale_image():
    start_time = time.time()
    initial_memory = get_memory_usage()

    try:
        # Check memory before processing
        if initial_memory > 8000:  # 8GB limit
            force_memory_cleanup()
            if get_memory_usage() > 8000:
                return jsonify({'error': 'Memory usage too high, please try again later'}), 503

        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']
        image = Image.open(image_file.stream).convert('RGB')

        # Limit image size to prevent memory explosion
        max_pixels = 2048 * 2048  # 4MP limit
        if image.width * image.height > max_pixels:
            scale_factor = (max_pixels / (image.width * image.height)) ** 0.5
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"Resized image to {new_width}x{new_height} to prevent memory issues")

        # Convert to tensor
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)

        # Process with tiling using configured tile size
        with torch.no_grad():
            output_tensor = process_image_tiled(image_tensor, tile_size=config_tile_size, tile_pad=16)

        if output_tensor is None:
            return jsonify({'error': 'Failed to process image'}), 500

        # Convert back to PIL
        output_tensor = torch.clamp(output_tensor, 0, 1)
        output_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_np = (output_np * 255).astype(np.uint8)
        output_image = Image.fromarray(output_np)

        # Clean up tensors immediately
        del image_tensor, output_tensor
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Save to buffer
        buffer = io.BytesIO()
        output_image.save(buffer, format='JPEG', quality=95, optimize=True)
        buffer.seek(0)

        processing_time = int((time.time() - start_time) * 1000)
        final_memory = get_memory_usage()

        print(f"[INFO] Processing completed in {processing_time}ms")
        print(f"[INFO] Memory: {initial_memory:.1f}MB -> {final_memory:.1f}MB")

        # Force cleanup after processing
        force_memory_cleanup()

        response = send_file(buffer, mimetype='image/jpeg')
        response.headers['X-Processing-Time'] = str(processing_time)
        return response

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[ERROR] Failed to process image")
        print(f"[ERROR] Error type: {type(e).__name__}")
        print(f"[ERROR] Error message: {e}")
        print(f"[ERROR] Full traceback:")
        print(error_details)
        print("=" * 50)
        # Clean up on error
        if device and device.type == 'cuda':
            torch.cuda.empty_cache()
        force_memory_cleanup()
        return jsonify({'error': f'Processing failed: {str(e)}', 'details': error_details}), 500

if __name__ == '__main__':
    print("[START] Starting Memory-Optimized Real-ESRGAN Service...")
    print(f"[INFO] Initial memory usage: {get_memory_usage():.1f} MB")

    # Initialize model at startup
    if not initialize_model():
        print("[ERROR] Failed to initialize model, exiting...")
        exit(1)

    print(f"[READY] Service ready. Memory usage: {get_memory_usage():.1f} MB")
    print("[INFO] Port: 8083")
    print("[INFO] Health check: http://localhost:8083/health")

    app.run(host='0.0.0.0', port=8083, debug=True, threaded=True)