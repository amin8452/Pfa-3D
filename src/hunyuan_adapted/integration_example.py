"""
Example of how to integrate the adapted Hunyuan3D-2 model
"""

import torch
from .adapter import load_hunyuan_model

# Example usage
def example_usage():
    # Load the adapted model
    model = load_hunyuan_model(
        checkpoint_path='checkpoints/hunyuan3d_base.pth',
        latent_dim=512,
        num_points=2048
    )

    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Set to evaluation mode
    model.eval()

    # Example input (batch of 1 image)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # Forward pass
    with torch.no_grad():
        points = model(dummy_input)

    print(f"Output point cloud shape: {points.shape}")

    # Fine-tune the model
    model.fine_tune(freeze_hunyuan=True)

    # Now only the adapter layers will be updated during training
    # The Hunyuan3D-2 parameters are frozen

    return model

if __name__ == '__main__':
    example_usage()
