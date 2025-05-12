"""
Utility functions for the Hunyuan3D-Glasses project
"""

import os
import torch
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2


def ensure_directory(directory):
    """
    Ensure that a directory exists
    """
    os.makedirs(directory, exist_ok=True)
    return directory


def save_checkpoint(model, optimizer, epoch, metrics, output_dir, is_best=False):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        metrics: Evaluation metrics
        output_dir: Output directory
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # Ensure directory exists
    ensure_directory(output_dir)
    
    # Save latest checkpoint
    checkpoint_path = os.path.join(output_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint if this is the best model
    if is_best:
        best_path = os.path.join(output_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
    
    # Save epoch checkpoint
    epoch_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, epoch_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, device='cuda'):
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        device: Device to load model onto
    
    Returns:
        model: Model with loaded weights
        optimizer: Optimizer with loaded state (if provided)
        epoch: Epoch of the checkpoint
        metrics: Metrics from the checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    return model, optimizer, epoch, metrics


def visualize_point_cloud(points, title='Point Cloud', save_path=None):
    """
    Visualize a point cloud using matplotlib
    
    Args:
        points: Point cloud tensor (N, 3)
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    # Convert to numpy if tensor
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.max([
        np.ptp(points[:, 0]),
        np.ptp(points[:, 1]),
        np.ptp(points[:, 2])
    ])
    mid_x = np.mean([np.min(points[:, 0]), np.max(points[:, 0])])
    mid_y = np.mean([np.min(points[:, 1]), np.max(points[:, 1])])
    mid_z = np.mean([np.min(points[:, 2]), np.max(points[:, 2])])
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def point_cloud_to_mesh(points, output_path=None):
    """
    Convert a point cloud to a mesh using Poisson reconstruction
    
    Args:
        points: Point cloud tensor (N, 3)
        output_path: Path to save the mesh (optional)
    
    Returns:
        mesh: Trimesh mesh object
    """
    # Convert to numpy if tensor
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    
    # Create a mesh using marching cubes
    mesh = trimesh.voxel.ops.points_to_marching_cubes(
        points, pitch=0.05, radius=0.02
    )
    
    # Save if output path is provided
    if output_path:
        mesh.export(output_path)
    
    return mesh


def preprocess_image(image_path, img_size=224):
    """
    Preprocess an image for the model
    
    Args:
        image_path: Path to the image
        img_size: Size to resize the image to
    
    Returns:
        img_tensor: Preprocessed image tensor (1, C, H, W)
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
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    
    return img_tensor
