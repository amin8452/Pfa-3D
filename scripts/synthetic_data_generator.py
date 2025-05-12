"""
Script to generate synthetic data for training the Hunyuan3D-Glasses model
"""

import os
import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import random


def load_base_models(models_dir):
    """
    Load base 3D glasses models
    
    Args:
        models_dir: Directory containing base models
    
    Returns:
        list: List of loaded models
    """
    models = []
    
    # Check if directory exists
    if not os.path.exists(models_dir):
        print(f"Warning: Directory {models_dir} does not exist. No base models loaded.")
        return models
    
    # Load all .obj files in the directory
    for filename in os.listdir(models_dir):
        if filename.endswith('.obj'):
            try:
                model_path = os.path.join(models_dir, filename)
                mesh = trimesh.load(model_path)
                models.append({
                    'mesh': mesh,
                    'name': os.path.splitext(filename)[0]
                })
                print(f"Loaded model: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return models


def create_variations(base_model, num_variations=10):
    """
    Create variations of a base model by applying random transformations
    
    Args:
        base_model: Base model to create variations from
        num_variations: Number of variations to create
    
    Returns:
        list: List of model variations
    """
    variations = []
    
    for i in range(num_variations):
        # Create a copy of the base mesh
        mesh = base_model['mesh'].copy()
        
        # Apply random scaling
        scale_factor = 0.9 + 0.2 * random.random()  # 0.9 to 1.1
        mesh.apply_scale(scale_factor)
        
        # Apply random rotation (small angles)
        rotation = trimesh.transformations.rotation_matrix(
            angle=np.radians(random.uniform(-15, 15)),
            direction=[0, 1, 0],
            point=mesh.centroid
        )
        mesh.apply_transform(rotation)
        
        # Apply small random noise to vertices
        noise_scale = 0.01
        noise = np.random.normal(0, noise_scale, mesh.vertices.shape)
        mesh.vertices += noise
        
        # Add to variations
        variations.append({
            'mesh': mesh,
            'name': f"{base_model['name']}_var_{i}",
            'base_name': base_model['name']
        })
    
    return variations


def render_model(mesh, resolution=(224, 224), angles=None):
    """
    Render a 3D model from different angles
    
    Args:
        mesh: Trimesh mesh to render
        resolution: Image resolution
        angles: List of angles to render from (if None, use default angles)
    
    Returns:
        list: List of rendered images
    """
    if angles is None:
        # Default angles: front, side, and 45 degrees
        angles = [
            (0, 0, 0),      # Front view
            (0, np.pi/2, 0), # Side view
            (0, np.pi/4, 0), # 45 degrees
            (np.pi/8, np.pi/4, 0),  # Slightly from above
            (np.pi/8, -np.pi/4, 0)  # Slightly from above, other side
        ]
    
    images = []
    
    # Create a scene
    scene = pyrender.Scene()
    
    # Add mesh to scene
    mesh_render = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_render)
    
    # Add light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light)
    
    # Create camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    
    # Render from different angles
    for rx, ry, rz in angles:
        # Position camera based on angle
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = trimesh.transformations.euler_matrix(rx, ry, rz)[:3, :3]
        camera_pose[2, 3] = 2.0  # Distance from origin
        
        # Look at center
        camera_pose = trimesh.transformations.look_at(
            camera_pose[:3, 3],
            [0, 0, 0],
            [0, 1, 0]
        )
        
        # Add camera to scene
        camera_node = scene.add(camera, pose=camera_pose)
        
        # Render
        r = pyrender.OffscreenRenderer(resolution[0], resolution[1])
        color, depth = r.render(scene)
        
        # Convert to RGB
        color = color.astype(np.uint8)
        
        # Add to images
        images.append(color)
        
        # Remove camera for next iteration
        scene.remove_node(camera_node)
        
        # Clean up
        r.delete()
    
    return images


def generate_dataset(num_samples=1000, output_dir='data', base_models_dir='base_models'):
    """
    Generate a synthetic dataset for training
    
    Args:
        num_samples: Total number of samples to generate
        output_dir: Output directory
        base_models_dir: Directory containing base models
    """
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'models'), exist_ok=True)
    
    # Load base models
    base_models = load_base_models(base_models_dir)
    
    # If no base models found, create some primitive glasses shapes
    if not base_models:
        print("No base models found. Creating primitive glasses shapes...")
        base_models = create_primitive_glasses()
    
    # Create variations of base models
    all_models = []
    for base_model in base_models:
        variations = create_variations(base_model, num_variations=max(1, num_samples // len(base_models) // 5))
        all_models.extend(variations)
    
    # Shuffle models
    random.shuffle(all_models)
    
    # Split into train, val, test
    train_ratio, val_ratio = 0.8, 0.1
    train_size = int(len(all_models) * train_ratio)
    val_size = int(len(all_models) * val_ratio)
    
    train_models = all_models[:train_size]
    val_models = all_models[train_size:train_size+val_size]
    test_models = all_models[train_size+val_size:]
    
    # Generate dataset
    for split, models in [('train', train_models), ('val', val_models), ('test', test_models)]:
        print(f"Generating {split} set...")
        for i, model in enumerate(tqdm(models)):
            # Generate multiple views per model
            images = render_model(model['mesh'])
            
            # Save model
            model_path = os.path.join(output_dir, split, 'models', f"{model['name']}.obj")
            model['mesh'].export(model_path)
            
            # Save images
            for j, img in enumerate(images):
                img_path = os.path.join(output_dir, split, 'images', f"{model['name']}_view_{j}.jpg")
                cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print(f"Dataset generation complete. Generated {len(all_models)} models with multiple views.")


def create_primitive_glasses():
    """
    Create primitive glasses shapes when no base models are available
    
    Returns:
        list: List of primitive glasses models
    """
    models = []
    
    # Create a simple glasses frame
    def create_glasses_frame():
        # Create left lens (circle)
        left_lens = trimesh.creation.annulus(r_min=0.3, r_max=0.4, height=0.05)
        left_lens.apply_translation([-0.5, 0, 0])
        
        # Create right lens (circle)
        right_lens = trimesh.creation.annulus(r_min=0.3, r_max=0.4, height=0.05)
        right_lens.apply_translation([0.5, 0, 0])
        
        # Create bridge (cylinder)
        bridge = trimesh.creation.cylinder(radius=0.05, height=0.5)
        bridge.apply_translation([0, 0, 0])
        bridge.apply_rotation(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
        
        # Create temples (arms)
        left_temple = trimesh.creation.cylinder(radius=0.05, height=1.0)
        left_temple.apply_translation([-0.5, 0, -0.2])
        left_temple.apply_rotation(trimesh.transformations.rotation_matrix(np.pi/2, [0, 0, 1]))
        
        right_temple = trimesh.creation.cylinder(radius=0.05, height=1.0)
        right_temple.apply_translation([0.5, 0, -0.2])
        right_temple.apply_rotation(trimesh.transformations.rotation_matrix(np.pi/2, [0, 0, 1]))
        
        # Combine all parts
        glasses = trimesh.util.concatenate([left_lens, right_lens, bridge, left_temple, right_temple])
        
        return glasses
    
    # Create a few variations
    for i in range(5):
        glasses = create_glasses_frame()
        
        # Apply some random transformations
        scale = 0.8 + 0.4 * random.random()
        glasses.apply_scale(scale)
        
        models.append({
            'mesh': glasses,
            'name': f"primitive_glasses_{i}"
        })
    
    return models


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic dataset for glasses reconstruction')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory')
    parser.add_argument('--base_models_dir', type=str, default='base_models',
                        help='Directory containing base models')
    
    args = parser.parse_args()
    
    generate_dataset(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        base_models_dir=args.base_models_dir
    )
