"""
Script to download pre-trained models for Hunyuan3D-Glasses
"""

import os
import argparse
import requests
from tqdm import tqdm
import hashlib
import subprocess
import sys
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Download pre-trained models')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Directory to save models')
    parser.add_argument('--model', type=str, default='all',
                        choices=['base', 'finetuned', 'all', 'repo'],
                        help='Which model to download (repo: clone the entire repository)')
    parser.add_argument('--hunyuan_dir', type=str, default='Hunyuan3D-2',
                        help='Directory to clone the Hunyuan3D-2 repository')
    return parser.parse_args()


def download_file(url, output_path, expected_md5=None):
    """
    Download a file with progress bar
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Check if file already exists and has correct MD5
    if os.path.exists(output_path) and expected_md5:
        with open(output_path, 'rb') as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        if md5 == expected_md5:
            print(f"File already exists and has correct MD5: {output_path}")
            return

    # Download the file
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Get file size
    total_size = int(response.headers.get('content-length', 0))

    # Download with progress bar
    with open(output_path, 'wb') as f, tqdm(
        desc=os.path.basename(output_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    # Verify MD5 if provided
    if expected_md5:
        with open(output_path, 'rb') as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        if md5 != expected_md5:
            os.remove(output_path)
            raise ValueError(f"MD5 mismatch for {output_path}. Expected {expected_md5}, got {md5}")
        else:
            print(f"MD5 verified: {md5}")


def clone_hunyuan3d_repo(output_dir):
    """
    Clone the Hunyuan3D-2 repository from Tencent
    """
    repo_url = "https://github.com/Tencent/Hunyuan3D-2.git"

    # Check if directory already exists
    if os.path.exists(output_dir):
        print(f"Directory {output_dir} already exists. Pulling latest changes...")
        try:
            # Pull latest changes if it's a git repository
            cmd = f"cd {output_dir} && git pull"
            subprocess.run(cmd, shell=True, check=True)
            print("Successfully updated repository.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error updating repository: {e}")
            return False

    # Clone the repository
    print(f"Cloning Hunyuan3D-2 repository to {output_dir}...")
    try:
        cmd = f"git clone {repo_url} {output_dir}"
        subprocess.run(cmd, shell=True, check=True)
        print("Successfully cloned repository.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        return False


def find_model_files(hunyuan_dir):
    """
    Find model files in the Hunyuan3D-2 repository

    Returns:
        dict: Dictionary of model files found
    """
    model_files = {}

    # Check if the repository directory exists
    if not os.path.exists(hunyuan_dir):
        return model_files

    # Common locations for model files in ML repositories
    potential_locations = [
        os.path.join(hunyuan_dir, 'checkpoints'),
        os.path.join(hunyuan_dir, 'pretrained'),
        os.path.join(hunyuan_dir, 'weights'),
        os.path.join(hunyuan_dir, 'models'),
        hunyuan_dir
    ]

    # Common model file extensions
    model_extensions = ['.pth', '.pt', '.ckpt', '.bin', '.weights']

    # Search for model files
    for location in potential_locations:
        if os.path.exists(location) and os.path.isdir(location):
            for root, _, files in os.walk(location):
                for file in files:
                    if any(file.endswith(ext) for ext in model_extensions):
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB

                        # Only consider files larger than 10MB as potential model files
                        if file_size > 10:
                            # Determine model type based on filename
                            model_type = 'base'
                            if 'fine' in file.lower() or 'tuned' in file.lower():
                                model_type = 'finetuned'

                            # Add to model files
                            if model_type not in model_files:
                                model_files[model_type] = []

                            model_files[model_type].append({
                                'path': file_path,
                                'size': file_size,
                                'name': file
                            })

    return model_files


def copy_model_file(source_path, target_path):
    """
    Copy a model file from source to target
    """
    try:
        # Create target directory if it doesn't exist
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # Copy the file
        shutil.copy2(source_path, target_path)
        print(f"Copied model file from {source_path} to {target_path}")
        return True
    except Exception as e:
        print(f"Error copying model file: {e}")
        return False


def main():
    args = parse_args()

    # Check if we should clone the entire repository
    if args.model == 'repo':
        success = clone_hunyuan3d_repo(args.hunyuan_dir)
        if success:
            print(f"Hunyuan3D-2 repository cloned to {args.hunyuan_dir}")

            # Find model files in the cloned repository
            model_files = find_model_files(args.hunyuan_dir)

            if model_files:
                print("\nFound the following model files in the repository:")
                for model_type, files in model_files.items():
                    print(f"\n{model_type.capitalize()} models:")
                    for i, file_info in enumerate(files):
                        print(f"  {i+1}. {file_info['name']} ({file_info['size']:.2f} MB)")

                print("\nYou can use these models directly or copy them to your checkpoints directory.")
                print("To copy them, run: python scripts/download_pretrained.py --model all")
            else:
                print("\nNo model files found in the repository.")
                print("The repository might not include pre-trained weights.")
                print("Check the repository's README for instructions on how to obtain pre-trained weights.")
        else:
            print("Failed to clone repository. Please check your internet connection and try again.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Find model files in the cloned repository
    model_files = find_model_files(args.hunyuan_dir)

    # Define models to download from Hugging Face
    # Currently only the base model URL is working reliably
    models = {
        'base': {
            'url': 'https://huggingface.co/tencent/Hunyuan3D-2/resolve/main/hunyuan3d-dit-v2-0/model.safetensors',
            'output_path': os.path.join(args.output_dir, 'hunyuan3d_base.safetensors'),
            'md5': None  # Add MD5 when available
        }
        # Other models are commented out as they were not found at the specified URLs
        # Uncomment if URLs become available
        # 'finetuned': {
        #     'url': 'https://huggingface.co/tencent/Hunyuan3D-2/resolve/main/hunyuan3d-paint-v2-0/model.safetensors',
        #     'output_path': os.path.join(args.output_dir, 'hunyuan3d_paint.safetensors'),
        #     'md5': None
        # },
        # 'mini': {
        #     'url': 'https://huggingface.co/tencent/Hunyuan3D-2mini/resolve/main/hunyuan3d-dit-v2-mini/model.safetensors',
        #     'output_path': os.path.join(args.output_dir, 'hunyuan3d_mini.safetensors'),
        #     'md5': None
        # },
        # 'turbo': {
        #     'url': 'https://huggingface.co/tencent/Hunyuan3D-2/resolve/main/hunyuan3d-dit-v2-0-turbo/model.safetensors',
        #     'output_path': os.path.join(args.output_dir, 'hunyuan3d_turbo.safetensors'),
        #     'md5': None
        # }
    }

    # Determine which models to download
    if args.model == 'all':
        models_to_download = list(models.keys())
    else:
        models_to_download = [args.model]

    # Process models
    for model_name in models_to_download:
        model_info = models[model_name]

        # Check if model file exists in the cloned repository
        if model_name in model_files and model_files[model_name]:
            # Use the first model file found
            source_path = model_files[model_name][0]['path']
            print(f"Found {model_name} model in the cloned repository: {source_path}")

            # Copy the model file
            if copy_model_file(source_path, model_info['output_path']):
                print(f"Successfully copied {model_name} model to {model_info['output_path']}")
                continue

        # If model file not found in repository or copy failed, try downloading
        print(f"Downloading {model_name} model...")
        try:
            download_file(
                model_info['url'],
                model_info['output_path'],
                model_info['md5']
            )
            print(f"Successfully downloaded {model_name} model to {model_info['output_path']}")
        except Exception as e:
            print(f"Error downloading {model_name} model: {e}")
            print(f"You may want to try cloning the entire repository with: python scripts/download_pretrained.py --model repo")
            print(f"Then check the repository's README for instructions on how to obtain pre-trained weights.")

    print("Download complete!")


if __name__ == '__main__':
    main()
