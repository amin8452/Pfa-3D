"""
Script to download pre-trained models for Hunyuan3D-Glasses
"""

import os
import argparse
import requests
from tqdm import tqdm
import hashlib


def parse_args():
    parser = argparse.ArgumentParser(description='Download pre-trained models')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Directory to save models')
    parser.add_argument('--model', type=str, default='all',
                        choices=['base', 'finetuned', 'all'],
                        help='Which model to download')
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


def main():
    args = parse_args()
    
    # Define models to download
    models = {
        'base': {
            'url': 'https://github.com/amin8452/Hunyuan3D-Glasses/releases/download/v1.0/hunyuan3d_base.pth',
            'output_path': os.path.join(args.output_dir, 'hunyuan3d_base.pth'),
            'md5': None  # Add MD5 when available
        },
        'finetuned': {
            'url': 'https://github.com/amin8452/Hunyuan3D-Glasses/releases/download/v1.0/hunyuan3d_glasses.pth',
            'output_path': os.path.join(args.output_dir, 'hunyuan3d_glasses.pth'),
            'md5': None  # Add MD5 when available
        }
    }
    
    # Determine which models to download
    if args.model == 'all':
        models_to_download = list(models.keys())
    else:
        models_to_download = [args.model]
    
    # Download models
    for model_name in models_to_download:
        model_info = models[model_name]
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
    
    print("Download complete!")


if __name__ == '__main__':
    main()
