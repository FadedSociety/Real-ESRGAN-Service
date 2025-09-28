#!/usr/bin/env python3
"""
Simple Real-ESRGAN Setup Script
Installs dependencies and downloads the model file
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def install_package(package):
    """Install a Python package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package}")
        return False

def install_pytorch_cuda():
    """Install PyTorch with CUDA support"""
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ])
        print("✓ Installed PyTorch with CUDA support")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install PyTorch with CUDA")
        return False

def ask_gpu_support():
    """Ask user if they have NVIDIA GPU"""
    print("GPU Detection")
    print("=============")
    print("Do you have an NVIDIA GPU for faster AI processing?")
    print("1. Yes - Install CUDA version (recommended for NVIDIA GPUs)")
    print("2. No - Install CPU version (works on all computers)")
    print()

    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == "1":
            return True
        elif choice == "2":
            return False
        else:
            print("Please enter 1 or 2")

def download_model():
    """Download the Real-ESRGAN model file"""
    model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    model_path = "RealESRGAN_x4plus_anime_6B.pth"
    models_dir_path = os.path.join("models", "RealESRGAN_x4plus_anime_6B.pth")

    # Check if model already exists in either location
    if os.path.exists(model_path):
        print(f"✓ Model already exists: {model_path}")
        return True
    elif os.path.exists(models_dir_path):
        print(f"✓ Model already exists: {models_dir_path}")
        return True

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    print(f"Downloading model from {model_url}...")
    print("This may take a few minutes (file is ~6MB)...")

    try:
        urllib.request.urlretrieve(model_url, models_dir_path)
        print(f"✓ Model downloaded to: {models_dir_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        return False

def main():
    print("Real-ESRGAN Simple Setup")
    print("=" * 30)
    print()

    # Ask user about GPU support
    has_gpu = ask_gpu_support()

    # Install PyTorch based on GPU choice
    if has_gpu:
        print("\nInstalling PyTorch with CUDA support...")
        torch_success = install_pytorch_cuda()
    else:
        print("\nInstalling PyTorch (CPU version)...")
        torch_success = install_package("torch") and install_package("torchvision")

    # Required packages (excluding torch/torchvision since we install them separately)
    packages = [
        "pillow",
        "opencv-python",
        "flask",
        "psutil",
        "numpy"
    ]

    print("\nInstalling other Python packages...")
    failed_packages = []

    if not torch_success:
        failed_packages.extend(["torch", "torchvision"])

    for package in packages:
        if not install_package(package):
            failed_packages.append(package)

    if failed_packages:
        print(f"\n⚠️ Failed to install: {', '.join(failed_packages)}")
        print("You may need to install these manually:")
        for pkg in failed_packages:
            print(f"  pip install {pkg}")
        print()
    else:
        print("\n✓ All packages installed successfully!")

    print("\nDownloading model file...")
    if download_model():
        print("\n✅ Setup completed successfully!")
        print("\nYou can now run: python memory_optimized_real_esrgan_app.py")
        if has_gpu:
            print("🚀 GPU acceleration enabled - AI processing will be much faster!")
        else:
            print("💻 CPU mode - processing will be slower but still works")
    else:
        print("\n❌ Setup failed!")
        print("You may need to download the model manually from:")
        print("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth")

if __name__ == "__main__":
    main()