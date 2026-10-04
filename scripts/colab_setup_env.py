#!/usr/bin/env python3
"""
Colab GPU Automated Environment Setup for EditCTC
Installs system dependencies, PaddlePaddle GPU (CUDA 12.3/13.0 compatible),
and project requirements for high-performance training & inference.
"""

import os
import sys
import subprocess
import time

def run_cmd(cmd, check=True):
    print(f"[*] Running: {cmd}")
    res = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if res.stdout:
        print(res.stdout)
    if res.stderr and res.returncode != 0:
        print(f"[!] Stderr: {res.stderr}")
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with code {res.returncode}: {cmd}")
    return res

def setup_environment():
    start_time = time.time()
    print("==========================================================")
    print("🚀 EditCTC Colab Environment Automated Setup")
    print("==========================================================")

    # 1. System packages
    print("\n[1/5] Installing system packages (aria2, libnvtoolsext1)...")
    run_cmd("apt-get update -qq && apt-get install -y -qq aria2 libnvtoolsext1", check=False)

    # 2. Check Paddle GPU Wheel
    whl_url = "https://paddle-whl.cdn.bcebos.com/stable/cu123/paddlepaddle-gpu/paddlepaddle_gpu-3.1.0-cp313-cp313-linux_x86_64.whl"
    whl_path = "/tmp/paddlepaddle_gpu-3.1.0-cp313-cp313-linux_x86_64.whl"

    print("\n[2/5] Checking/Downloading PaddlePaddle GPU wheel (1.6GB)...")
    if not os.path.exists(whl_path) or os.path.getsize(whl_path) < 1600 * 1024 * 1024:
        print("[*] Downloading with aria2c 16 connections...")
        run_cmd(f"aria2c -x 16 -s 16 -k 1M -d /tmp -o {os.path.basename(whl_path)} --allow-overwrite=true {whl_url}")
    else:
        print(f"[+] Found cached wheel: {whl_path} ({os.path.getsize(whl_path)/(1024*1024):.2f} MB)")

    # 3. Clean and Install Paddle GPU
    print("\n[3/5] Installing PaddlePaddle GPU and python dependencies...")
    run_cmd(f"{sys.executable} -m pip uninstall -y paddlepaddle paddlepaddle-gpu", check=False)
    run_cmd(f"{sys.executable} -m pip install {whl_path} --no-deps", check=True)
    run_cmd(f"{sys.executable} -m pip install opt_einsum protobuf Pillow safetensors httpx pyyaml visualdl opencv-python tqdm albumentations", check=False)

    # 4. Verification
    print("\n[4/5] Verifying GPU and PaddlePaddle CUDA...")
    verify_code = """
import paddle
print(f'PaddlePaddle Version: {paddle.__version__}')
print(f'CUDA Compiled: {paddle.is_compiled_with_cuda()}')
print(f'Current Device: {paddle.device.get_device()}')
if not paddle.is_compiled_with_cuda():
    print('WARNING: Paddle is not using CUDA!')
    exit(1)
"""
    res = subprocess.run([sys.executable, "-c", verify_code], text=True, capture_output=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f"[!] Verification failed: {res.stderr}")
    else:
        print("[+] PaddlePaddle GPU verified successfully!")

    # 5. Check Drive Mount & Paths
    print("\n[5/5] Checking Google Drive mount...")
    drive_data_path = "/content/drive/MyDrive/research/EditCTC/release_EditCTC"
    if os.path.exists(drive_data_path):
        print(f"[+] Found Drive project directory: {drive_data_path}")
        print("    Contents:", os.listdir(drive_data_path))
    else:
        print(f"[!] Warning: Drive project not found at {drive_data_path}. Ensure Google Drive is mounted.")

    print("\n==========================================================")
    print(f"✨ Setup completed in {time.time() - start_time:.2f} seconds!")
    print("==========================================================")

if __name__ == "__main__":
    setup_environment()
