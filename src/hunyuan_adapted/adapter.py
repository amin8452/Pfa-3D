"""
Adapter module to integrate Hunyuan3D-2 with our glasses reconstruction model
"""

import os
import torch
import torch.nn as nn
import numpy as np

from .network import Network
from .renderer import Renderer


class Hunyuan3DAdapter(nn.Module):
    """
    Adapter class for Hunyuan3D-2 model

    This class adapts the Hunyuan3D-2 model for glasses reconstruction by:
    1. Loading the pre-trained Hunyuan3D-2 model
    2. Adapting the input/output interfaces to match our glasses reconstruction pipeline
    3. Providing fine-tuning capabilities
    """

    def __init__(self, checkpoint_path=None, latent_dim=512, num_points=2048):
        super(Hunyuan3DAdapter, self).__init__()

        self.latent_dim = latent_dim
        self.num_points = num_points

        # Create Hunyuan3D-2 network
        self.network = Network()

        # Create renderer
        self.renderer = Renderer()

        # Create adapter layers
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, latent_dim)
        )

        self.point_decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_points * 3)
        )

        # Load checkpoint if provided
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        """
        Load pre-trained weights from checkpoint
        """
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint {checkpoint_path} not found. Using random initialization.")
            return

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load Hunyuan3D-2 weights
        if 'network' in checkpoint:
            self.network.load_state_dict(checkpoint['network'])
            print(f"Loaded Hunyuan3D-2 network weights from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint {checkpoint_path} does not contain network weights.")

        # Load renderer weights if available
        if 'renderer' in checkpoint:
            self.renderer.load_state_dict(checkpoint['renderer'])
            print(f"Loaded Hunyuan3D-2 renderer weights from {checkpoint_path}")

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input image tensor (B, C, H, W)

        Returns:
            points: 3D point cloud (B, num_points, 3)
        """
        # Encode image to latent representation
        latent = self.image_encoder(x)

        # Decode latent to point cloud
        points_flat = self.point_decoder(latent)
        points = points_flat.view(-1, self.num_points, 3)

        return points

    def fine_tune(self, freeze_hunyuan=True):
        """
        Prepare model for fine-tuning

        Args:
            freeze_hunyuan: If True, freeze the Hunyuan3D-2 parameters
        """
        # Freeze Hunyuan3D-2 parameters if requested
        if freeze_hunyuan:
            for param in self.network.parameters():
                param.requires_grad = False

            for param in self.renderer.parameters():
                param.requires_grad = False

        # Always train the adapter layers
        for param in self.image_encoder.parameters():
            param.requires_grad = True

        for param in self.point_decoder.parameters():
            param.requires_grad = True


def load_hunyuan_model(checkpoint_path=None, latent_dim=512, num_points=2048):
    """
    Load the Hunyuan3D-2 model with our adapter

    Args:
        checkpoint_path: Path to Hunyuan3D-2 checkpoint
        latent_dim: Dimension of latent space
        num_points: Number of points in output point cloud

    Returns:
        model: Adapted Hunyuan3D-2 model
    """
    model = Hunyuan3DAdapter(
        checkpoint_path=checkpoint_path,
        latent_dim=latent_dim,
        num_points=num_points
    )

    return model
