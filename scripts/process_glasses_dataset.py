"""
Script to process the specific glasses dataset with 600+ images
"""

import os
import sys
import argparse
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

# Add the src directory to the path so we can import the adapter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hunyuan_adapted.adapter import load_hunyuan_model


def parse_args():
    parser = argparse.ArgumentParser(description='Process the glasses dataset')
    parser.add_argument('--dataset_path', type=str, default='data/train/images',
                        help='Path to the dataset of images')
    parser.add_argument('--output_dir', type=str, default='results/meshes',
                        help='Directory to save the generated meshes')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/hunyuan3d_base.safetensors',
                        help='Path to the model checkpoint')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for processing')
    parser.add_argument('--num_images', type=int, default=None,
                        help='Number of images to process (None for all)')
    parser.add_argument('--start_index', type=int, default=0,
                        help='Index to start processing from')
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


def process_dataset(args):
    """
    Process the dataset and generate 3D models
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
    
    # Get list of image files
    image_files = [f for f in os.listdir(args.dataset_path) 
                  if any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp'])]
    
    # Sort the files for consistent processing
    image_files.sort()
    
    # Limit the number of images if specified
    if args.num_images is not None:
        end_index = min(args.start_index + args.num_images, len(image_files))
        image_files = image_files[args.start_index:end_index]
    else:
        image_files = image_files[args.start_index:]
    
    print(f"Processing {len(image_files)} images starting from index {args.start_index}")
    
    # Load the model
    print(f"Loading model from {args.checkpoint}...")
    try:
        model = load_hunyuan_model(
            checkpoint_path=args.checkpoint,
            latent_dim=512,
            num_points=2048
        )
        
        # Set the model to evaluation mode
        model.eval()
        
        # Move the model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"Using device: {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create a directory for visualizations if needed
    if args.visualize:
        vis_dir = os.path.join(args.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
    
    # Process each image
    start_time = time.time()
    processed_count = 0
    error_count = 0
    
    with torch.no_grad():
        for i, image_file in enumerate(tqdm(image_files, desc="Processing images")):
            try:
                # Get full path to image
                image_path = os.path.join(args.dataset_path, image_file)
                
                # Get output path
                output_name = os.path.splitext(image_file)[0] + ".obj"
                output_path = os.path.join(args.output_dir, output_name)
                
                # Skip if the output file already exists
                if os.path.exists(output_path):
                    print(f"Skipping {image_file} as {output_path} already exists.")
                    continue
                
                # Process the image
                points = model(image_path)
                
                # Generate a mesh from the points
                mesh = model.generate_mesh(points)
                
                # Save the mesh
                mesh.export(output_path)
                processed_count += 1
                
                # Visualize if requested
                if args.visualize:
                    visualize_results(image_path, points, vis_dir, image_file)
                
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                error_count += 1
    
    # Calculate processing time and statistics
    total_time = time.time() - start_time
    avg_time_per_image = total_time / max(1, processed_count)
    
    print(f"\nProcessing complete!")
    print(f"Total images processed: {processed_count}")
    print(f"Errors encountered: {error_count}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per image: {avg_time_per_image:.2f} seconds")
    print(f"Results saved to {args.output_dir}")


def visualize_results(image_path, points, vis_dir, image_file):
    """
    Visualize the input image and the generated 3D model
    """
    # Create a figure with two subplots
    fig = plt.figure(figsize=(12, 6))
    
    # Plot the input image
    ax1 = fig.add_subplot(121)
    img = Image.open(image_path)
    ax1.imshow(np.array(img))
    ax1.set_title('Input Image')
    ax1.axis('off')
    
    # Plot the 3D model
    ax2 = fig.add_subplot(122, projection='3d')
    points_np = points[0].cpu().numpy()
    ax2.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], s=1)
    ax2.set_title('Generated 3D Model')
    ax2.axis('off')
    
    # Set equal aspect ratio
    max_range = np.max([
        np.ptp(points_np[:, 0]),
        np.ptp(points_np[:, 1]),
        np.ptp(points_np[:, 2])
    ])
    mid_x = np.mean([np.min(points_np[:, 0]), np.max(points_np[:, 0])])
    mid_y = np.mean([np.min(points_np[:, 1]), np.max(points_np[:, 1])])
    mid_z = np.mean([np.min(points_np[:, 2]), np.max(points_np[:, 2])])
    ax2.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax2.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax2.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    # Save the figure
    vis_path = os.path.join(vis_dir, os.path.splitext(image_file)[0] + '.png')
    plt.savefig(vis_path)
    plt.close()


def main():
    args = parse_args()
    process_dataset(args)


if __name__ == '__main__':
    main()
