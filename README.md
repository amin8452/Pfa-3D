# Hunyuan3D-Glasses

A deep learning project for 3D glasses reconstruction from 2D images, based on the Hunyuan3D model.

## Overview

This project adapts the Hunyuan3D model for the specific task of reconstructing 3D glasses models from 2D images. It includes a complete pipeline for training, fine-tuning, evaluation, and visualization of 3D glasses reconstructions.

## Features

- 3D glasses reconstruction from single 2D images
- Fine-tuning capabilities for the Hunyuan3D model
- Comprehensive evaluation metrics (Chamfer distance, EMD, IoU)
- Visualization tools for 3D reconstruction results
- Training and validation pipelines
- Support for Google Colab and Kaggle environments

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA (for GPU acceleration)

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/amin8452/Pfa-3D.git
cd Pfa-3D
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained models (optional):
```bash
python scripts/download_pretrained.py
```

### Google Colab / Kaggle Setup

You can run this project on Google Colab or Kaggle without a local installation:

1. Open the notebook in Google Colab:
   - [Open Pfa-3D Colab Notebook](https://colab.research.google.com/github/amin8452/Pfa-3D/blob/master/notebooks/hunyuan3d_glasses_colab.ipynb)

2. Or use the provided setup script:
```python
# Clone the repository
!git clone https://github.com/amin8452/Pfa-3D.git
%cd Pfa-3D

# Set up the environment
!python scripts/setup_colab.py
```

## Dataset

The project uses a dataset of 2D glasses images paired with their 3D models.

### Dataset Structure

```
data/
├── train/
│   ├── images/
│   │   ├── glasses_001.jpg
│   │   ├── glasses_002.jpg
│   │   └── ...
│   └── models/
│       ├── glasses_001.obj
│       ├── glasses_002.obj
│       └── ...
├── val/
│   ├── images/
│   │   └── ...
│   └── models/
│       └── ...
└── test/
    ├── images/
    │   └── ...
    └── models/
        └── ...
```

### Preparing Your Own Dataset

To use your own dataset:

1. Organize your data following the structure above
2. Update the data paths in the configuration files

### Generating Synthetic Data

If you don't have a real dataset, you can generate synthetic data:

```bash
python scripts/synthetic_data_generator.py --num_samples 1000
```

This script will:
1. Create primitive 3D glasses models or use base models if provided
2. Generate variations of these models
3. Render the models from different angles to create 2D images
4. Save the data in the required directory structure

## Usage

### Training

To train the model from scratch:

```bash
python src/train_glasses.py --config configs/train_config.yaml
```

### Fine-tuning

To fine-tune a pre-trained model:

```bash
python src/train_glasses.py --config configs/finetune_config.yaml --checkpoint checkpoints/pretrained.pth
```

### Evaluation

To evaluate a trained model:

```bash
python src/evaluate_glasses.py --config configs/eval_config.yaml --checkpoint checkpoints/model.pth
```

### Inference

To reconstruct 3D glasses from a single image:

```bash
python src/inference.py --image path/to/image.jpg --output path/to/output.obj
```

## Results

The model achieves the following performance on the test set:

| Metric | Value |
|--------|-------|
| Chamfer Distance | 0.XX |
| EMD | 0.XX |
| IoU | 0.XX |

## Project Structure

```
Hunyuan3D-Glasses/
├── checkpoints/        # Saved model checkpoints
├── configs/            # Configuration files
├── data/               # Dataset
├── notebooks/          # Jupyter notebooks for visualization
├── results/            # Evaluation results and visualizations
├── scripts/            # Utility scripts
│   ├── create_data_dirs.py     # Create data directory structure
│   ├── download_pretrained.py  # Download pre-trained models
│   ├── setup_colab.py          # Set up environment on Colab/Kaggle
│   ├── setup_github.py         # Set up GitHub repository
│   └── synthetic_data_generator.py  # Generate synthetic dataset
├── src/                # Source code
│   ├── data_loader_glasses.py  # Data loading utilities
│   ├── evaluate_glasses.py     # Evaluation script
│   ├── inference.py            # Inference script
│   ├── metrics.py              # Evaluation metrics
│   ├── model.py                # Model architecture
│   ├── train_glasses.py        # Training script
│   └── utils.py                # Utility functions
└── README.md           # This file
```
## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Running on Colab/Kaggle

This project is designed to run on cloud platforms with limited resources. Here's how to use it on Google Colab or Kaggle:

### Google Colab

1. Open the provided notebook:
   - [Open Hunyuan3D-Glasses Colab Notebook](https://colab.research.google.com/github/amin8452/Hunyuan3D-Glasses/blob/master/notebooks/hunyuan3d_glasses_colab.ipynb)

2. Follow the instructions in the notebook to:
   - Set up the environment
   - Generate or upload a dataset
   - Train the model
   - Evaluate results
   - Perform inference

### Kaggle

1. Create a new notebook on Kaggle
2. Add the following code to get started:

```python
# Clone the repository
!git clone https://github.com/amin8452/Pfa-3D.git
%cd Pfa-3D

# Set up the environment
!python scripts/setup_colab.py --platform kaggle

# Generate synthetic data
!python scripts/synthetic_data_generator.py --num_samples 500

# Train the model (with reduced batch size for Kaggle's resources)
!python src/train_glasses.py --config configs/train_config.yaml --output_dir checkpoints/initial_training
```
## Acknowledgements

- This project adapts the Hunyuan3D model for glasses reconstruction
- Thanks to the original authors of Hunyuan3D for their foundational work
