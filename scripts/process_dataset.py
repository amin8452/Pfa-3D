"""
Script to process a dataset of glasses images and generate 3D models
"""

import os
import sys
import argparse
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the path so we can import the adapter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hunyuan_adapted.adapter import load_hunyuan_model


def parse_args():
    parser = argparse.ArgumentParser(description='Process a dataset of glasses images')
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
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the results')
    return parser.parse_args()


def find_images(dataset_path):
    """
    Find all image files in the dataset path
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = []
    
    # Check if dataset_path is a directory
    if os.path.isdir(dataset_path):
        # Walk through the directory and find all image files
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
    else:
        # Check if dataset_path is a file
        if os.path.isfile(dataset_path) and any(dataset_path.lower().endswith(ext) for ext in image_extensions):
            image_files.append(dataset_path)
    
    return image_files


def check_dataset(dataset_path):
    """
    Check if the dataset exists and contains images
    """
    # Check if the path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path {dataset_path} does not exist.")
        return False
    
    # Find all image files
    image_files = find_images(dataset_path)
    
    if not image_files:
        print(f"Error: No images found in {dataset_path}.")
        return False
    
    print(f"Found {len(image_files)} images in {dataset_path}.")
    print(f"Examples: {[os.path.basename(f) for f in image_files[:5]]}")
    
    return True


def process_dataset(args):
    """
    Process the dataset and generate 3D models
    """
    # Check if the dataset exists and contains images
    if not check_dataset(args.dataset_path):
        return
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all image files
    image_files = find_images(args.dataset_path)
    
    # Limit the number of images if specified
    if args.num_images is not None:
        image_files = image_files[:args.num_images]
    
    # Load the model
    print(f"Loading model from {args.checkpoint}...")
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
    
    # Process each image
    print(f"Processing {len(image_files)} images...")
    with torch.no_grad():
        for image_path in tqdm(image_files):
            try:
                # Get the output path
                output_name = os.path.splitext(os.path.basename(image_path))[0] + ".obj"
                output_path = os.path.join(args.output_dir, output_name)
                
                # Skip if the output file already exists
                if os.path.exists(output_path):
                    print(f"Skipping {image_path} as {output_path} already exists.")
                    continue
                
                # Process the image
                points = model(image_path)
                
                # Generate a mesh from the points
                mesh = model.generate_mesh(points)
                
                # Save the mesh
                mesh.export(output_path)
                
                # Visualize if requested
                if args.visualize:
                    visualize_results(image_path, points, output_path)
            
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    
    print(f"Processing complete. Results saved to {args.output_dir}")


def visualize_results(image_path, points, mesh_path):
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
    vis_dir = os.path.join(os.path.dirname(mesh_path), 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    vis_path = os.path.join(vis_dir, os.path.basename(mesh_path).replace('.obj', '.png'))
    plt.savefig(vis_path)
    plt.close()


def main():
    args = parse_args()
    process_dataset(args)


if __name__ == '__main__':
    main()
