"""
Evaluation metrics for 3D reconstruction
"""

import torch
import torch.nn.functional as F
import numpy as np
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points


def chamfer_loss(pred_points, gt_points, batch_reduction='mean'):
    """
    Compute Chamfer distance between two point clouds

    Args:
        pred_points: Predicted point cloud (B, N, 3)
        gt_points: Ground truth point cloud (B, M, 3)
        batch_reduction: Reduction method for batch dimension ('mean', 'sum', or None)

    Returns:
        loss: Chamfer distance
    """
    # Compute bidirectional chamfer distance
    loss, _ = chamfer_distance(pred_points, gt_points, batch_reduction=batch_reduction)

    return loss


def f1_score(pred_points, gt_points, threshold=0.01):
    """
    Compute F1 score between two point clouds

    Args:
        pred_points: Predicted point cloud (B, N, 3)
        gt_points: Ground truth point cloud (B, M, 3)
        threshold: Distance threshold for considering a point as correctly predicted

    Returns:
        f1: F1 score
        precision: Precision
        recall: Recall
    """
    batch_size = pred_points.shape[0]

    # Compute nearest neighbors from prediction to ground truth
    nn_pred = knn_points(pred_points, gt_points, K=1)
    # Compute nearest neighbors from ground truth to prediction
    nn_gt = knn_points(gt_points, pred_points, K=1)

    # Get distances
    dist_pred_to_gt = nn_pred.dists[..., 0]  # (B, N)
    dist_gt_to_pred = nn_gt.dists[..., 0]  # (B, M)

    # Compute precision and recall
    precision = torch.mean((dist_pred_to_gt < threshold).float(), dim=1)  # (B,)
    recall = torch.mean((dist_gt_to_pred < threshold).float(), dim=1)  # (B,)

    # Compute F1 score
    f1 = 2 * precision * recall / (precision + recall + 1e-8)  # (B,)

    # Average over batch
    return f1.mean().item(), precision.mean().item(), recall.mean().item()


def normal_consistency(pred_normals, gt_normals, pred_points, gt_points):
    """
    Compute normal consistency between two point clouds

    Args:
        pred_normals: Predicted normals (B, N, 3)
        gt_normals: Ground truth normals (B, M, 3)
        pred_points: Predicted point cloud (B, N, 3)
        gt_points: Ground truth point cloud (B, M, 3)

    Returns:
        consistency: Normal consistency score
    """
    batch_size = pred_points.shape[0]

    # Normalize normals
    pred_normals = F.normalize(pred_normals, dim=2)
    gt_normals = F.normalize(gt_normals, dim=2)

    # Find nearest neighbors from prediction to ground truth
    nn_pred = knn_points(pred_points, gt_points, K=1)
    idx = nn_pred.idx  # (B, N, 1)

    # Gather corresponding ground truth normals
    batch_indices = torch.arange(batch_size, device=pred_points.device).view(-1, 1, 1).expand(-1, idx.shape[1], -1)
    gt_normals_nn = gt_normals[batch_indices, idx, :]  # (B, N, 1, 3)
    gt_normals_nn = gt_normals_nn.squeeze(2)  # (B, N, 3)

    # Compute consistency as absolute dot product
    consistency = torch.abs(torch.sum(pred_normals * gt_normals_nn, dim=2))  # (B, N)

    # Average over points and batch
    return consistency.mean().item()


def compute_all_metrics(pred_points, gt_points, pred_normals=None, gt_normals=None):
    """
    Compute all metrics for evaluation

    Args:
        pred_points: Predicted point cloud (B, N, 3)
        gt_points: Ground truth point cloud (B, M, 3)
        pred_normals: Predicted normals (B, N, 3) or None
        gt_normals: Ground truth normals (B, M, 3) or None

    Returns:
        metrics: Dictionary of metrics
    """
    metrics = {}

    # Chamfer distance
    metrics['chamfer_distance'] = chamfer_loss(pred_points, gt_points).item()

    # F1 score
    f1, precision, recall = f1_score(pred_points, gt_points)
    metrics['f1_score'] = f1
    metrics['precision'] = precision
    metrics['recall'] = recall

    # Normal consistency (if normals are provided)
    if pred_normals is not None and gt_normals is not None:
        metrics['normal_consistency'] = normal_consistency(pred_normals, gt_normals, pred_points, gt_points)

    return metrics
