"""
Script to set up the environment on Google Colab or Kaggle
"""

import os
import subprocess
import sys


def run_command(command):
    """Run a shell command and print output"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(result.stdout)
    return True


def setup_environment(platform="colab"):
    """
    Set up the environment on Google Colab or Kaggle
    
    Args:
        platform: 'colab' or 'kaggle'
    """
    print(f"Setting up environment for {platform}...")
    
    # Install basic dependencies
    run_command("pip install torch torchvision")
    run_command("pip install trimesh numpy scipy matplotlib opencv-python pyyaml tqdm tensorboard plotly requests")
    
    # Install PyTorch3D (this can be complex, so we use a pre-built version)
    import torch
    pyt_version_str = torch.__version__.split("+")[0].replace(".", "")
    version_str = "".join([
        f"py3{sys.version_info.minor}_cu",
        torch.version.cuda.replace(".", ""),
        f"_pyt{pyt_version_str}"
    ])
    run_command("pip install fvcore iopath")
    run_command(f"pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html")
    
    # Install additional packages for synthetic data generation
    run_command("pip install pyrender")
    
    # Create data directory structure
    if not os.path.exists("data"):
        os.makedirs("data", exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join("data", split, "images"), exist_ok=True)
        os.makedirs(os.path.join("data", split, "models"), exist_ok=True)
    
    # Create output directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    print("Environment setup complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Set up environment on Colab or Kaggle")
    parser.add_argument("--platform", choices=["colab", "kaggle"], default="colab",
                        help="Platform to set up for")
    
    args = parser.parse_args()
    
    setup_environment(args.platform)
