"""
Evaluation script with 3D metrics for glasses reconstruction
"""

import os
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import trimesh
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene

from model import Hunyuan3DGlasses
from data_loader_glasses import get_dataloader
from metrics import compute_all_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Hunyuan3D model for glasses reconstruction')
    parser.add_argument('--config', type=str, default='configs/eval_config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results')
    parser.add_argument('--save_meshes', action='store_true',
                        help='Save reconstructed meshes')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def visualize_results(pred_points, gt_points, sample_id, output_dir):
    """
    Visualize predicted and ground truth point clouds
    """
    # Create a scene dictionary for plotly
    scene_dict = {
        'Predicted': Pointclouds(points=[pred_points]),
        'Ground Truth': Pointclouds(points=[gt_points])
    }

    # Create the plot
    fig = plot_scene(scene_dict)

    # Save the plot
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    fig.write_html(os.path.join(output_dir, 'visualizations', f'{sample_id}.html'))


def save_mesh(points, sample_id, output_dir):
    """
    Convert point cloud to mesh and save as OBJ file
    """
    # Convert points to numpy
    points_np = points.cpu().numpy()

    # Create a point cloud
    point_cloud = trimesh.points.PointCloud(points_np)

    # Create a mesh using Poisson reconstruction
    # This is a simple approximation - more advanced methods could be used
    mesh = trimesh.voxel.ops.points_to_marching_cubes(
        points_np, pitch=0.05, radius=0.02
    )

    # Save the mesh
    os.makedirs(os.path.join(output_dir, 'meshes'), exist_ok=True)
    mesh.export(os.path.join(output_dir, 'meshes', f'{sample_id}.obj'))


def evaluate(model, data_loader, device, output_dir, visualize=False, save_meshes=False):
    """
    Evaluate the model on the dataset
    """
    model.eval()
    all_metrics = {}
    all_sample_metrics = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluation'):
            # Get data
            images = batch['image'].to(device)
            points = batch['points'].to(device)
            sample_ids = batch['sample_id']

            # Forward pass
            pred_points = model(images)

            # Compute metrics
            batch_metrics = compute_all_metrics(pred_points, points)

            # Process each sample in the batch
            for i in range(len(sample_ids)):
                sample_id = sample_ids[i]
                pred_points_i = pred_points[i].unsqueeze(0)
                gt_points_i = points[i].unsqueeze(0)

                # Compute metrics for this sample
                sample_metrics = compute_all_metrics(pred_points_i, gt_points_i)
                all_sample_metrics[sample_id] = sample_metrics

                # Visualize if requested
                if visualize:
                    visualize_results(
                        pred_points_i[0], gt_points_i[0],
                        sample_id, output_dir
                    )

                # Save mesh if requested
                if save_meshes:
                    save_mesh(pred_points_i[0], sample_id, output_dir)

            # Accumulate batch metrics
            for k, v in batch_metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v)

    # Average metrics
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}

    return avg_metrics, all_sample_metrics


def main():
    # Parse arguments
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create data loader
    data_loader = get_dataloader(
        data_dir=config['data']['data_dir'],
        split=config['data']['split'],
        batch_size=config['evaluation']['batch_size'],
        num_workers=config['data']['num_workers'],
        img_size=config['data']['img_size'],
        num_points=config['data']['num_points'],
        augment=False
    )

    # Create model
    model = Hunyuan3DGlasses(
        latent_dim=config['model']['latent_dim'],
        num_points=config['data']['num_points']
    )

    # Load checkpoint
    print(f'Loading checkpoint from {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Evaluate
    print(f'Evaluating on {config["data"]["split"]} set...')
    avg_metrics, sample_metrics = evaluate(
        model, data_loader, device, args.output_dir,
        visualize=args.visualize, save_meshes=args.save_meshes
    )

    # Print metrics
    print('Average metrics:')
    for k, v in avg_metrics.items():
        print(f'  {k} = {v:.6f}')

    # Save metrics
    with open(os.path.join(args.output_dir, 'metrics.yaml'), 'w') as f:
        yaml.dump({'average': avg_metrics, 'samples': sample_metrics}, f)

    # Plot metrics distribution
    for metric_name in avg_metrics.keys():
        values = [metrics[metric_name] for metrics in sample_metrics.values()]
        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=20)
        plt.title(f'Distribution of {metric_name}')
        plt.xlabel(metric_name)
        plt.ylabel('Count')
        plt.savefig(os.path.join(args.output_dir, f'{metric_name}_distribution.png'))
        plt.close()

    print(f'Evaluation complete. Results saved to {args.output_dir}')


if __name__ == '__main__':
    main()
