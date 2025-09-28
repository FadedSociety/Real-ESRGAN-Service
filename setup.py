#!/usr/bin/env python3
"""
Real-ESRGAN Setup Script
Automatically installs dependencies and downloads required models
"""

import subprocess
import sys
import os
import urllib.request
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors gracefully"""
    print(f"[INSTALL] {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"[SUCCESS] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("[CHECK] Checking Python version...")
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"[SUCCESS] Python {sys.version.split()[0]} detected")
    return True

def check_nvidia_gpu():
    """Check if NVIDIA GPU is available"""
    print("[CHECK] Checking for NVIDIA GPU...")
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("[SUCCESS] NVIDIA GPU detected")
            return True
    except:
        pass
    print("[WARNING] NVIDIA GPU not detected - will use CPU (much slower)")
    return False

def install_pytorch_cuda():
    """Install PyTorch with CUDA support"""
    print("[CHECK] Checking PyTorch installation...")

    # Check if PyTorch is already installed with CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[SUCCESS] PyTorch with CUDA already installed (version {torch.__version__})")
            return True
        else:
            print("[WARNING] PyTorch found but without CUDA support")
    except ImportError:
        print("[INFO] PyTorch not found")

    # Uninstall existing PyTorch first
    print("[INFO] Removing existing PyTorch installation...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"],
                  capture_output=True)

    # Install CUDA version
    cuda_command = [
        sys.executable, "-m", "pip", "install", "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ]

    if run_command(" ".join(cuda_command), "Installing PyTorch with CUDA support"):
        # Verify installation
        try:
            import torch
            if torch.cuda.is_available():
                print(f"[SUCCESS] PyTorch with CUDA successfully installed")
                print(f"   GPU detected: {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'Unknown'}")
                return True
        except:
            pass

    print("[WARNING] CUDA installation failed, falling back to CPU version")
    cpu_command = [sys.executable, "-m", "pip", "install", "torch", "torchvision"]
    return run_command(" ".join(cpu_command), "Installing PyTorch (CPU version)")

def install_dependencies():
    """Install required Python packages"""
    dependencies = [
        "flask",
        "pillow",
        "opencv-python",
        "psutil",
        "numpy",
        "gradio"
    ]

    for dep in dependencies:
        if not run_command(f"{sys.executable} -m pip install {dep}", f"Installing {dep}"):
            return False
    return True

def download_model():
    """Download Real-ESRGAN model if not exists"""
    model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    model_path = Path("RealESRGAN_x4plus_anime_6B.pth")

    if model_path.exists():
        print(f"[SUCCESS] Model already exists: {model_path}")
        return True

    print("[DOWNLOAD] Downloading Real-ESRGAN model (this may take a few minutes)...")
    try:
        urllib.request.urlretrieve(model_url, model_path)
        print(f"[SUCCESS] Model downloaded successfully: {model_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Model download failed: {e}")
        return False

def create_start_script():
    """Create a simple start script for the service"""
    script_content = '''#!/usr/bin/env python3
"""
Real-ESRGAN Service Launcher
Runs setup check and starts the service
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("[START] Starting Real-ESRGAN Service...")

    # Quick dependency check
    try:
        import torch
        import flask
        import cv2
        import PIL
        print("[SUCCESS] All dependencies found")
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("[INFO] Run setup.py first to install dependencies")
        return

    # Check model file
    model_path = Path("RealESRGAN_x4plus_anime_6B.pth")
    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        print("[INFO] Run setup.py first to download the model")
        return

    # Start the service
    try:
        print("[INFO] You can now start the service with:")
        print("   python gradio_real_esrgan_app.py  (Full web interface with file browser)")
        print("   python memory_optimized_real_esrgan_app.py  (API only)")
        print("[INFO] Gradio version includes file browsing for model selection")
    except Exception as e:
        print(f"[ERROR] Failed to start service: {e}")

if __name__ == "__main__":
    main()
'''

    with open("start_service.py", "w") as f:
        f.write(script_content)

    print("[SUCCESS] Created start_service.py launcher")

def main():
    """Main setup function"""
    print("Real-ESRGAN for Suwayomi Setup")
    print("=" * 50)

    # Check requirements
    if not check_python_version():
        return False

    has_gpu = check_nvidia_gpu()

    # Install dependencies
    print("\n[INSTALL] Installing Dependencies...")
    print("-" * 30)

    if not install_dependencies():
        print("[ERROR] Failed to install basic dependencies")
        return False

    if has_gpu:
        if not install_pytorch_cuda():
            print("[ERROR] Failed to install PyTorch with CUDA")
            return False
    else:
        # Install CPU version
        if not run_command(f"{sys.executable} -m pip install torch torchvision", "Installing PyTorch (CPU version)"):
            return False

    # Download model
    print("\n[DOWNLOAD] Downloading Model...")
    print("-" * 20)

    if not download_model():
        print("[ERROR] Failed to download model")
        return False

    # Create launcher script
    create_start_script()

    # Final verification
    print("\n[VERIFY] Final Verification...")
    print("-" * 20)

    try:
        import torch
        gpu_status = "[SUCCESS] CUDA available" if torch.cuda.is_available() else "[WARNING] CPU only"
        print(f"PyTorch: {torch.__version__} ({gpu_status})")

        import flask
        print(f"Flask: {flask.__version__}")

        model_path = Path("RealESRGAN_x4plus_anime_6B.pth")
        model_size = model_path.stat().st_size / (1024*1024) if model_path.exists() else 0
        print(f"Model: {model_size:.1f}MB")

    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
        return False

    print("\n" + "=" * 50)
    print("[SUCCESS] Setup completed successfully!")
    print()
    print("Next steps:")
    print("1. Start the service: python start_service.py")
    print("2. Start Suwayomi server")
    print("3. Configure Suwayomi to use: http://localhost:8083/upscale")
    print()
    print("Service will be available at: http://localhost:8083")
    print("Health check: http://localhost:8083/health")

    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n[ERROR] Setup failed - please check the errors above")
        sys.exit(1)
    else:
        print("\n[SUCCESS] Setup complete!")