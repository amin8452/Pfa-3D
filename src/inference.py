"""
Inference script for 3D glasses reconstruction from a single image
"""

import os
import argparse
import torch
import numpy as np
import cv2
import trimesh
import matplotlib.pyplot as plt
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene

from model import Hunyuan3DGlasses


def parse_args():
    parser = argparse.ArgumentParser(description='Reconstruct 3D glasses from a single image')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='output.obj',
                        help='Path to output 3D model')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the reconstruction')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Size to resize input image to')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='Number of points in the output point cloud')
    parser.add_argument('--latent_dim', type=int, default=512,
                        help='Dimension of latent space')
    return parser.parse_args()


def preprocess_image(image_path, img_size):
    """
    Preprocess the input image
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, (img_size, img_size))
    
    # Convert to tensor and normalize
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    
    return img


def points_to_mesh(points, output_path):
    """
    Convert point cloud to mesh and save as OBJ file
    """
    # Convert points to numpy
    points_np = points.cpu().numpy()
    
    # Create a mesh using Poisson reconstruction
    # This is a simple approximation - more advanced methods could be used
    mesh = trimesh.voxel.ops.points_to_marching_cubes(
        points_np, pitch=0.05, radius=0.02
    )
    
    # Save the mesh
    mesh.export(output_path)
    
    return mesh


def visualize_reconstruction(points, output_dir):
    """
    Visualize the reconstructed point cloud
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    
    # Create a scene dictionary for plotly
    scene_dict = {
        'Reconstructed': Pointclouds(points=[points])
    }
    
    # Create the plot
    fig = plot_scene(scene_dict)
    
    # Save the plot
    html_path = os.path.splitext(output_dir)[0] + '.html'
    fig.write_html(html_path)
    print(f"Visualization saved to {html_path}")
    
    # Also create a simple matplotlib plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    points_np = points.cpu().numpy()
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], s=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Reconstructed 3D Glasses')
    
    # Set equal aspect ratio
    max_range = np.max([
        np.ptp(points_np[:, 0]),
        np.ptp(points_np[:, 1]),
        np.ptp(points_np[:, 2])
    ])
    mid_x = np.mean([np.min(points_np[:, 0]), np.max(points_np[:, 0])])
    mid_y = np.mean([np.min(points_np[:, 1]), np.max(points_np[:, 1])])
    mid_z = np.mean([np.min(points_np[:, 2]), np.max(points_np[:, 2])])
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    # Save the plot
    plt_path = os.path.splitext(output_dir)[0] + '.png'
    plt.savefig(plt_path)
    plt.close()
    print(f"Plot saved to {plt_path}")


def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model
    model = Hunyuan3DGlasses(
        latent_dim=args.latent_dim,
        num_points=args.num_points
    )
    
    # Load checkpoint
    print(f'Loading checkpoint from {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Preprocess image
    print(f'Processing image: {args.image}')
    img = preprocess_image(args.image, args.img_size)
    img = img.to(device)
    
    # Perform inference
    print('Reconstructing 3D model...')
    with torch.no_grad():
        pred_points = model(img)[0]  # (N, 3)
    
    # Convert to mesh and save
    print(f'Saving 3D model to {args.output}')
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    mesh = points_to_mesh(pred_points, args.output)
    
    # Visualize if requested
    if args.visualize:
        print('Visualizing reconstruction...')
        visualize_reconstruction(pred_points, args.output)
    
    print('Done!')


if __name__ == '__main__':
    main()
