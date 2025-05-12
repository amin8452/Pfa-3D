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

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA (for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/amin8452/Hunyuan3D-Glasses.git
cd Hunyuan3D-Glasses
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained models (optional):
```bash
python scripts/download_pretrained.py
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
2. Update the data paths in `config.py`

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

## Acknowledgements

- This project adapts the Hunyuan3D model for glasses reconstruction
- Thanks to the original authors of Hunyuan3D for their foundational work
