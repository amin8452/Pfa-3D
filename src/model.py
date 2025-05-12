"""
Hunyuan3D model adapted for glasses reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class ImageEncoder(nn.Module):
    """
    Encoder network that processes 2D images and extracts features
    """
    def __init__(self, in_channels=3, latent_dim=512):
        super(ImageEncoder, self).__init__()
        
        # ResNet-like architecture
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling and projection to latent space
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, latent_dim)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # First block with potential downsampling
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class ResidualBlock(nn.Module):
    """
    Basic residual block for the encoder
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class GlassesDecoder(nn.Module):
    """
    Decoder network that generates 3D glasses from latent representation
    """
    def __init__(self, latent_dim=512, num_points=2048):
        super(GlassesDecoder, self).__init__()
        
        # MLP to process latent vector
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        
        # Transformer for point cloud generation
        encoder_layers = TransformerEncoderLayer(d_model=1024, nhead=8, dim_feedforward=2048)
        self.transformer = TransformerEncoder(encoder_layers, num_layers=3)
        
        # Final layers to generate point coordinates
        self.point_gen = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3 * num_points)  # x, y, z coordinates for each point
        )
        
        self.num_points = num_points
        
    def forward(self, x):
        # Process latent vector
        x = self.mlp(x)
        
        # Reshape for transformer
        x = x.unsqueeze(0)  # Add sequence dimension
        x = self.transformer(x)
        x = x.squeeze(0)  # Remove sequence dimension
        
        # Generate point cloud
        points = self.point_gen(x)
        points = points.view(-1, self.num_points, 3)
        
        return points


class Hunyuan3DGlasses(nn.Module):
    """
    Complete model for 3D glasses reconstruction from 2D images
    """
    def __init__(self, latent_dim=512, num_points=2048):
        super(Hunyuan3DGlasses, self).__init__()
        
        self.encoder = ImageEncoder(in_channels=3, latent_dim=latent_dim)
        self.decoder = GlassesDecoder(latent_dim=latent_dim, num_points=num_points)
        
    def forward(self, x):
        # Encode image to latent representation
        latent = self.encoder(x)
        
        # Decode latent representation to 3D point cloud
        points = self.decoder(latent)
        
        return points
    
    def fine_tune(self, freeze_encoder=True):
        """
        Prepare model for fine-tuning
        
        Args:
            freeze_encoder: If True, freeze the encoder parameters
        """
        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            # Unfreeze the last layer of encoder for fine-tuning
            for param in self.encoder.fc.parameters():
                param.requires_grad = True
        
        # Always train the decoder
        for param in self.decoder.parameters():
            param.requires_grad = True
