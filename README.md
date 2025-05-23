# Pfa-3D: 3D Glasses Reconstruction

A deep learning project for reconstructing 3D glasses models from 2D images, adapting the Tencent Hunyuan3D-2 model.

<p align="center">
  <img src="https://github.com/user-attachments/assets/efb402a1-0b09-41e0-a6cb-259d442e76aa" width="600">
</p>

## Overview

This project adapts the powerful Hunyuan3D-2 model from Tencent for the specific task of reconstructing 3D glasses models from 2D images. The system takes a single 2D image of glasses as input and generates a detailed 3D model, ready for visualization or 3D printing.

## Key Features

- **Single-image 3D Reconstruction**: Generate 3D glasses models from a single 2D image
- **Hunyuan3D-2 Integration**: Leverages Tencent's state-of-the-art 3D generation model
- **Fine-tuning Capabilities**: Adapt the model for specific types of glasses
- **Colab/Kaggle Support**: Run on cloud platforms with limited resources
- **Comprehensive Evaluation**: Metrics for assessing reconstruction quality

## Quick Start

### Google Colab (Recommended)

The easiest way to try this project is through Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amin8452/Pfa-3D/blob/master/notebooks/hunyuan3d_glasses_colab.ipynb)

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/amin8452/Pfa-3D.git
cd Pfa-3D
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download and adapt the Hunyuan3D-2 model:
```bash
# Clone the Hunyuan3D-2 repository
python scripts/download_pretrained.py --model repo

# Download pre-trained models from Hugging Face
python scripts/download_pretrained.py --model all

# Adapt the model for glasses reconstruction
# Note: Use the correct directory name where the repository was cloned
python scripts/adapt_hunyuan_model.py --hunyuan_dir Hunyuan3D-2
```

## Using the Hunyuan3D-2 Model

### Basic Usage

```python
from src.hunyuan_adapted.adapter import load_hunyuan_model

# Load the adapted model
model = load_hunyuan_model(
    checkpoint_path='checkpoints/hunyuan3d_base.safetensors',
    latent_dim=512,
    num_points=2048
)

# Process an image
points = model('path/to/glasses_image.jpg')

# Generate a mesh from points
mesh = model.generate_mesh(points)
mesh.export('glasses_3d.obj')
```

### Processing a Dataset of Images

If you have a dataset of glasses images (e.g., 600+ images), you can use our specialized dataset processing scripts:

```bash
# Process all images in the dataset
python scripts/process_glasses_dataset.py --dataset_path data/train/images --output_dir results/meshes

# Process with visualization
python scripts/process_glasses_dataset.py --dataset_path data/train/images --output_dir results/meshes --visualize

# Process a limited number of images
python scripts/process_glasses_dataset.py --dataset_path data/train/images --output_dir results/meshes --num_images 10

# Process from a specific starting point
python scripts/process_glasses_dataset.py --dataset_path data/train/images --output_dir results/meshes --start_index 100
```

### Batch Processing for Large Datasets

For large datasets, you can process images in batches to manage memory usage:

```bash
# Process in batches of 50 images
python scripts/batch_process.py --dataset_path data/train/images --output_dir results/meshes --batch_size 50

# Process in batches with visualization
python scripts/batch_process.py --dataset_path data/train/images --output_dir results/meshes --batch_size 50 --visualize
```

### Fine-tuning for Specific Glasses Types

```python
# Prepare the model for fine-tuning
model.fine_tune(freeze_hunyuan=True)

# Now only the adapter layers will be updated during training
# The Hunyuan3D-2 parameters are frozen
```

### Using the Lightweight Model

For environments with limited resources (like Colab or Kaggle):

```python
# Load the mini model
model = load_hunyuan_model(
    checkpoint_path='checkpoints/hunyuan3d_mini.safetensors',
    latent_dim=512,
    num_points=2048
)
```

## Training and Evaluation

### Training Pipeline

```bash
# Train the model with your glasses dataset
python src/train_glasses.py --config configs/train_config.yaml

# Fine-tune the pre-trained model
python src/train_glasses.py --config configs/finetune_config.yaml --checkpoint checkpoints/hunyuan3d_base.safetensors
```

### Evaluation

```bash
# Evaluate the model's performance
python src/evaluate_glasses.py --config configs/eval_config.yaml --checkpoint checkpoints/model.pth --visualize
```

### Inference

```bash
# Generate a 3D model from a single image
python src/inference.py --image path/to/glasses_image.jpg --output glasses_3d.obj
```

## Running on Cloud Platforms

### Google Colab

We provide two Colab notebooks:

1. **General Usage**: [Hunyuan3D Glasses Colab](https://colab.research.google.com/github/amin8452/Pfa-3D/blob/master/notebooks/hunyuan3d_glasses_colab.ipynb)
   - Set up the environment
   - Download and adapt the Hunyuan3D-2 model
   - Process individual glasses images

2. **Dataset Processing**: [Process Glasses Dataset](https://colab.research.google.com/github/amin8452/Pfa-3D/blob/master/notebooks/process_glasses_dataset.ipynb)
   - Specifically designed for processing large datasets (e.g., 600 images)
   - Upload your dataset as a zip file or individual images
   - Process all images and download the results

### Kaggle

```python
# Clone the repository
!git clone https://github.com/amin8452/Pfa-3D.git
%cd Pfa-3D

# Set up the environment
!python scripts/setup_colab.py --platform kaggle

# Download and adapt the model
!python scripts/download_pretrained.py --model repo
!python scripts/download_pretrained.py --model all
# Note: Use the correct directory name where the repository was cloned
!python scripts/adapt_hunyuan_model.py --hunyuan_dir Hunyuan3D-2
```

## Project Structure

```
Pfa-3D/
├── checkpoints/        # Saved model checkpoints
├── configs/            # Configuration files
├── notebooks/          # Jupyter notebooks for Colab/Kaggle
├── scripts/            # Utility scripts
├── src/                # Source code
│   ├── hunyuan_adapted/  # Adapted Hunyuan3D-2 model
│   ├── data_loader_glasses.py  # Data loading utilities
│   ├── evaluate_glasses.py     # Evaluation script
│   ├── inference.py            # Inference script
│   ├── metrics.py              # Evaluation metrics
│   ├── model.py                # Model architecture
│   └── train_glasses.py        # Training script
└── README.md           # This file
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- This project adapts the [Hunyuan3D-2 model from Tencent](https://github.com/Tencent/Hunyuan3D-2) for glasses reconstruction
- Thanks to the original authors of Hunyuan3D-2 for their foundational work
