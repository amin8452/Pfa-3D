"""
Data loader for glasses dataset
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import trimesh
from pytorch3d.structures import Pointclouds
import random


class GlassesDataset(Dataset):
    """
    Dataset for 3D glasses reconstruction

    Args:
        data_dir: Root directory of the dataset
        split: 'train', 'val', or 'test'
        img_size: Size to resize images to
        num_points: Number of points to sample from 3D models
        transform: Optional transforms to apply to images
        augment: Whether to apply data augmentation
    """
    def __init__(self, data_dir, split='train', img_size=224, num_points=2048,
                 transform=None, augment=True):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.num_points = num_points
        self.transform = transform
        self.augment = augment and split == 'train'

        # Set up paths
        self.img_dir = os.path.join(data_dir, split, 'images')
        self.model_dir = os.path.join(data_dir, split, 'models')

        # Get list of samples
        self.samples = [f.split('.')[0] for f in os.listdir(self.img_dir)
                        if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]

        # Load image
        img_path = os.path.join(self.img_dir, f"{sample_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, f"{sample_id}.png")

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Apply augmentation if needed
        if self.augment:
            img = self._augment_image(img)

        # Convert to tensor and normalize
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

        # Apply additional transforms if provided
        if self.transform:
            img = self.transform(img)

        # Load 3D model
        model_path = os.path.join(self.model_dir, f"{sample_id}.obj")
        points = self._load_and_sample_model(model_path)

        return {
            'image': img,
            'points': points,
            'sample_id': sample_id
        }

    def _load_and_sample_model(self, model_path):
        """
        Load 3D model and sample points from its surface
        """
        mesh = trimesh.load(model_path)

        # Sample points from the mesh surface
        points, _ = trimesh.sample.sample_surface(mesh, self.num_points)

        # Normalize to unit cube
        center = (mesh.bounds[0] + mesh.bounds[1]) / 2
        scale = np.max(mesh.bounds[1] - mesh.bounds[0])
        points = (points - center) / scale

        return torch.from_numpy(points.astype(np.float32))

    def _augment_image(self, img):
        """
        Apply data augmentation to image
        """
        # Random horizontal flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)

        # Random brightness and contrast adjustment
        alpha = 0.9 + random.random() * 0.2  # 0.9-1.1
        beta = -10 + random.random() * 20  # -10 to 10
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # Random rotation (small angles)
        angle = random.uniform(-10, 10)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))

        return img


def get_dataloader(data_dir, split='train', batch_size=32, num_workers=4, **kwargs):
    """
    Create a dataloader for the glasses dataset

    Args:
        data_dir: Root directory of the dataset
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of workers for data loading
        **kwargs: Additional arguments for GlassesDataset

    Returns:
        DataLoader: PyTorch DataLoader
    """
    dataset = GlassesDataset(data_dir, split=split, **kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )


def create_data_loaders(data_dir, batch_size=32, num_workers=4, **kwargs):
    """
    Create data loaders for train, validation, and test sets

    Args:
        data_dir: Root directory of the dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        **kwargs: Additional arguments for GlassesDataset

    Returns:
        dict: Dictionary containing train, val, and test data loaders
    """
    train_loader = get_dataloader(
        data_dir,
        split='train',
        batch_size=batch_size,
        num_workers=num_workers,
        augment=True,
        **kwargs
    )

    val_loader = get_dataloader(
        data_dir,
        split='val',
        batch_size=batch_size,
        num_workers=num_workers,
        augment=False,
        **kwargs
    )

    test_loader = get_dataloader(
        data_dir,
        split='test',
        batch_size=batch_size,
        num_workers=num_workers,
        augment=False,
        **kwargs
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
