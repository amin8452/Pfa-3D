"""
Script to process the glasses dataset in batches
"""

import os
import argparse
import subprocess
import time
import math


def parse_args():
    parser = argparse.ArgumentParser(description='Process the glasses dataset in batches')
    parser.add_argument('--dataset_path', type=str, default='data/train/images',
                        help='Path to the dataset of images')
    parser.add_argument('--output_dir', type=str, default='results/meshes',
                        help='Directory to save the generated meshes')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/hunyuan3d_base.safetensors',
                        help='Path to the model checkpoint')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Number of images to process in each batch')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the results')
    return parser.parse_args()


def count_images(dataset_path):
    """
    Count the number of images in the dataset
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    count = 0
    
    for file in os.listdir(dataset_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            count += 1
    
    return count


def process_in_batches(args):
    """
    Process the dataset in batches
    """
    # Check if the dataset path exists
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path {args.dataset_path} does not exist.")
        return
    
    # Count the number of images
    num_images = count_images(args.dataset_path)
    print(f"Found {num_images} images in {args.dataset_path}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Calculate the number of batches
    num_batches = math.ceil(num_images / args.batch_size)
    print(f"Processing {num_images} images in {num_batches} batches of {args.batch_size}")
    
    # Process each batch
    start_time = time.time()
    
    for batch_idx in range(num_batches):
        start_index = batch_idx * args.batch_size
        
        # Prepare the command
        cmd = [
            "python", "scripts/process_glasses_dataset.py",
            "--dataset_path", args.dataset_path,
            "--output_dir", args.output_dir,
            "--checkpoint", args.checkpoint,
            "--start_index", str(start_index),
            "--num_images", str(args.batch_size)
        ]
        
        if args.visualize:
            cmd.append("--visualize")
        
        # Print the batch information
        print(f"\n{'='*80}")
        print(f"Processing Batch {batch_idx+1}/{num_batches} (Images {start_index+1}-{min(start_index+args.batch_size, num_images)})")
        print(f"{'='*80}")
        
        # Run the command
        subprocess.run(cmd)
    
    # Calculate total processing time
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"All batches completed!")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Results saved to {args.output_dir}")
    print(f"{'='*80}")


def main():
    args = parse_args()
    process_in_batches(args)


if __name__ == '__main__':
    main()
