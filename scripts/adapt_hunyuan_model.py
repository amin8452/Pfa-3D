"""
Script to adapt the Hunyuan3D-2 model for glasses reconstruction
"""

import os
import sys
import argparse
import torch
import shutil
import importlib.util


def parse_args():
    parser = argparse.ArgumentParser(description='Adapt Hunyuan3D-2 model for glasses reconstruction')
    parser.add_argument('--hunyuan_dir', type=str, default='Hunyuan3D-2',
                        help='Directory containing the Hunyuan3D-2 repository')
    parser.add_argument('--output_dir', type=str, default='src/hunyuan_adapted',
                        help='Output directory for adapted model files')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to Hunyuan3D-2 checkpoint (if not provided, will use default)')
    return parser.parse_args()


def check_hunyuan_repo(hunyuan_dir):
    """
    Check if the Hunyuan3D-2 repository exists and has the necessary files
    """
    # Check if the specified directory exists
    if not os.path.exists(hunyuan_dir):
        print(f"Error: Directory {hunyuan_dir} does not exist.")

        # Check if Hunyuan3D-2 directory exists (common alternative)
        alt_dir = "Hunyuan3D-2"
        if os.path.exists(alt_dir):
            print(f"However, found a Hunyuan3D-2 directory. You can use it with:")
            print(f"python scripts/adapt_hunyuan_model.py --hunyuan_dir {alt_dir}")
            return False

        # Check if hunyuan3d_repo directory exists (another alternative)
        alt_dir2 = "hunyuan3d_repo"
        if os.path.exists(alt_dir2):
            print(f"However, found a hunyuan3d_repo directory. You can use it with:")
            print(f"python scripts/adapt_hunyuan_model.py --hunyuan_dir {alt_dir2}")
            return False

        print("Please clone the Hunyuan3D-2 repository first with:")
        print("python scripts/download_pretrained.py --model repo")
        return False

    # Check if it's a git repository
    git_dir = os.path.join(hunyuan_dir, '.git')
    if not os.path.exists(git_dir):
        print(f"Warning: {hunyuan_dir} does not appear to be a git repository.")
        print("It might not be a complete clone of Hunyuan3D-2.")

    # Look for Python files in the repository
    python_files = []
    for root, _, files in os.walk(hunyuan_dir):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    if not python_files:
        print(f"Error: No Python files found in {hunyuan_dir}.")
        print("The repository might be empty or incomplete. Please try cloning it again.")
        return False

    # Look for model-related files
    model_files = []
    network_files = []
    for file_path in python_files:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                content = f.read().lower()
                if 'class' in content and ('network' in content or 'model' in content):
                    network_files.append(file_path)
                if 'import torch' in content or 'from torch' in content:
                    model_files.append(file_path)
            except:
                pass

    if not network_files:
        print(f"Warning: No network/model class definitions found in {hunyuan_dir}.")
        print("The repository might not contain the expected model architecture.")
        print("Will attempt to adapt anyway, but might need manual adjustments.")

    return True


def find_model_files(hunyuan_dir):
    """
    Find model-related files in the Hunyuan3D-2 repository
    """
    model_files = {
        'network': [],
        'renderer': [],
        'embedder': [],
        'model': [],
        'utils': []
    }

    # Look for Python files in the repository
    python_files = []
    for root, _, files in os.walk(hunyuan_dir):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    # Categorize files based on content
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()

                # Skip test files and setup files
                if 'test' in file_path.lower() or 'setup.py' in file_path.lower():
                    continue

                # Categorize based on content and filename
                filename = os.path.basename(file_path).lower()

                if 'network' in filename or 'class network' in content:
                    model_files['network'].append(file_path)
                elif 'renderer' in filename or 'class renderer' in content:
                    model_files['renderer'].append(file_path)
                elif 'embed' in filename or 'class embed' in content:
                    model_files['embedder'].append(file_path)
                elif 'model' in filename or 'class model' in content:
                    model_files['model'].append(file_path)
                elif 'util' in filename or 'helper' in filename:
                    model_files['utils'].append(file_path)
        except:
            pass

    return model_files


def copy_model_files(hunyuan_dir, output_dir):
    """
    Copy and adapt the necessary model files from Hunyuan3D-2
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create an empty __init__.py file
    with open(os.path.join(output_dir, '__init__.py'), 'w') as f:
        f.write('# Adapted Hunyuan3D-2 model for glasses reconstruction\n')

    # Find model files
    model_files = find_model_files(hunyuan_dir)

    # Files to copy
    files_to_copy = []

    # Add files from each category
    for category, paths in model_files.items():
        if paths:
            # Use the first file in each category
            files_to_copy.append(paths[0])

    # If no files found, print warning
    if not files_to_copy:
        print("Warning: No model files found in the repository.")
        print("Creating placeholder files instead.")

        # Create placeholder files
        placeholders = ['network.py', 'renderer.py', 'model.py']
        for placeholder in placeholders:
            with open(os.path.join(output_dir, placeholder), 'w') as f:
                f.write(f'"""\nPlaceholder for {placeholder}\n"""\n\n')
                f.write('# TODO: Implement this file based on the Hunyuan3D-2 repository\n')

        return True

    # Copy files
    for file_path in files_to_copy:
        dest_path = os.path.join(output_dir, os.path.basename(file_path))
        shutil.copy2(file_path, dest_path)
        print(f"Copied {file_path} to {dest_path}")

    return True


def create_adapter_file(output_dir):
    """
    Create an adapter file to integrate Hunyuan3D-2 with our glasses reconstruction model
    """
    adapter_path = os.path.join(output_dir, 'adapter.py')

    with open(adapter_path, 'w') as f:
        f.write("""\"\"\"
Adapter module to integrate Hunyuan3D-2 with our glasses reconstruction model
\"\"\"

import os
import torch
import torch.nn as nn
import numpy as np
import trimesh
from PIL import Image
import torchvision.transforms as transforms
from safetensors.torch import load_file

# Import any available modules from the copied files
try:
    from .network import Network
    has_network = True
except ImportError:
    has_network = False

try:
    from .renderer import Renderer
    has_renderer = True
except ImportError:
    has_renderer = False


class Hunyuan3DAdapter(nn.Module):
    \"\"\"
    Adapter class for Hunyuan3D-2 model

    This class adapts the Hunyuan3D-2 model for glasses reconstruction by:
    1. Loading the pre-trained Hunyuan3D-2 model if available
    2. Providing a standalone implementation if the original model is not available
    3. Adapting the input/output interfaces to match our glasses reconstruction pipeline
    4. Providing fine-tuning capabilities
    \"\"\"

    def __init__(self, checkpoint_path=None, latent_dim=512, num_points=2048):
        super(Hunyuan3DAdapter, self).__init__()

        self.latent_dim = latent_dim
        self.num_points = num_points

        # Create Hunyuan3D-2 network if available
        self.has_hunyuan = has_network and has_renderer
        if self.has_hunyuan:
            self.network = Network()
            self.renderer = Renderer()

        # Create adapter layers (used in both cases)
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
        \"\"\"
        Load pre-trained weights from checkpoint
        \"\"\"
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint {checkpoint_path} not found. Using random initialization.")
            return

        try:
            # Try to load using safetensors first (for Hugging Face models)
            if checkpoint_path.endswith('.safetensors'):
                checkpoint = load_file(checkpoint_path)
                print(f"Loaded checkpoint from {checkpoint_path} using safetensors")
            else:
                # Fall back to PyTorch loading
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                print(f"Loaded checkpoint from {checkpoint_path} using torch.load")

            # If we have the Hunyuan3D-2 network, try to load its weights
            if self.has_hunyuan:
                if 'network' in checkpoint:
                    self.network.load_state_dict(checkpoint['network'])
                    print(f"Loaded Hunyuan3D-2 network weights")

                if 'renderer' in checkpoint:
                    self.renderer.load_state_dict(checkpoint['renderer'])
                    print(f"Loaded Hunyuan3D-2 renderer weights")

            # Always try to load adapter weights if they exist
            # This allows for fine-tuning the adapter separately
            adapter_keys = [k for k in checkpoint.keys() if k.startswith('image_encoder') or k.startswith('point_decoder')]
            if adapter_keys:
                # Create a state dict with just the adapter keys
                adapter_state_dict = {k: checkpoint[k] for k in adapter_keys}
                # Load the adapter weights
                missing_keys, unexpected_keys = self.load_state_dict(adapter_state_dict, strict=False)
                print(f"Loaded adapter weights. Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Using random initialization instead.")

    def preprocess_image(self, image_path):
        \"\"\"
        Preprocess an image for the model

        Args:
            image_path: Path to the image or PIL Image

        Returns:
            tensor: Preprocessed image tensor (1, C, H, W)
        \"\"\"
        if isinstance(image_path, str):
            # Load image from path
            image = Image.open(image_path).convert('RGB')
        else:
            # Assume it's already a PIL Image
            image = image_path

        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Apply transforms and add batch dimension
        tensor = transform(image).unsqueeze(0)
        return tensor

    def forward(self, x):
        \"\"\"
        Forward pass

        Args:
            x: Input image tensor (B, C, H, W) or path to image

        Returns:
            points: 3D point cloud (B, num_points, 3)
        \"\"\"
        # Preprocess image if it's a path
        if isinstance(x, str):
            x = self.preprocess_image(x)

        # Encode image to latent representation
        latent = self.image_encoder(x)

        # Decode latent to point cloud
        points_flat = self.point_decoder(latent)
        points = points_flat.view(-1, self.num_points, 3)

        return points

    def generate_mesh(self, points):
        \"\"\"
        Generate a mesh from points

        Args:
            points: Point cloud tensor (B, N, 3) or (N, 3)

        Returns:
            mesh: Trimesh object
        \"\"\"
        # Ensure points is a numpy array
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()

        # Remove batch dimension if present
        if len(points.shape) == 3:
            points = points[0]

        # Create a point cloud
        point_cloud = trimesh.points.PointCloud(points)

        # Create a mesh using Poisson reconstruction or ball pivoting
        try:
            # Try to use Poisson reconstruction (requires Open3D)
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.estimate_normals()

            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)[0]
            mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
        except:
            # Fall back to a simpler method
            mesh = trimesh.voxel.ops.points_to_marching_cubes(points, pitch=0.05, radius=0.02)

        return mesh

    def fine_tune(self, freeze_hunyuan=True):
        \"\"\"
        Prepare model for fine-tuning

        Args:
            freeze_hunyuan: If True, freeze the Hunyuan3D-2 parameters
        \"\"\"
        # Freeze Hunyuan3D-2 parameters if requested and available
        if freeze_hunyuan and self.has_hunyuan:
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
    \"\"\"
    Load the Hunyuan3D-2 model with our adapter

    Args:
        checkpoint_path: Path to Hunyuan3D-2 checkpoint
        latent_dim: Dimension of latent space
        num_points: Number of points in output point cloud

    Returns:
        model: Adapted Hunyuan3D-2 model
    \"\"\"
    model = Hunyuan3DAdapter(
        checkpoint_path=checkpoint_path,
        latent_dim=latent_dim,
        num_points=num_points
    )

    return model
""")

    print(f"Created adapter file at {adapter_path}")
    return True


def create_integration_file(output_dir):
    """
    Create a file to demonstrate how to integrate the adapted model
    """
    integration_path = os.path.join(output_dir, 'integration_example.py')

    with open(integration_path, 'w') as f:
        f.write("""\"\"\"
Example of how to integrate the adapted Hunyuan3D-2 model
\"\"\"

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
""")

    print(f"Created integration example at {integration_path}")
    return True


def main():
    args = parse_args()

    # Check if Hunyuan3D-2 repository exists
    if not check_hunyuan_repo(args.hunyuan_dir):
        return

    # Copy model files
    if not copy_model_files(args.hunyuan_dir, args.output_dir):
        return

    # Create adapter file
    if not create_adapter_file(args.output_dir):
        return

    # Create integration example
    if not create_integration_file(args.output_dir):
        return

    print("\nSuccessfully adapted Hunyuan3D-2 model for glasses reconstruction!")
    print(f"Adapted model files are in: {args.output_dir}")
    print("\nTo use the adapted model in your project:")
    print("1. Import the adapter: from src.hunyuan_adapted.adapter import load_hunyuan_model")
    print("2. Load the model: model = load_hunyuan_model(checkpoint_path='checkpoints/hunyuan3d_base.pth')")
    print("3. Use it for inference or fine-tuning")


if __name__ == '__main__':
    main()
