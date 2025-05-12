# Training Data for 3D Glasses Reconstruction

This directory contains the training data for the 3D glasses reconstruction model.

## Directory Structure

```
data/
├── train/
│   ├── images/       # 2D images of glasses
│   └── models/       # Corresponding 3D models (if available)
├── val/
│   ├── images/
│   └── models/
└── test/
    ├── images/
    └── models/
```

## Adding Training Images

To add training images:

1. Place your 2D glasses images in the `images/` directory
2. If available, place corresponding 3D models in the `models/` directory

## Image Requirements

- Images should be in JPG, PNG, or other common image formats
- Recommended resolution: at least 512x512 pixels
- Clear, well-lit images of glasses with minimal background clutter
- Variety of angles and styles for better generalization

## Using the Dataset

The dataset can be processed using the provided scripts:

```bash
python scripts/process_dataset.py --dataset_path data/train/images --output_dir results/meshes
```

Or using the Colab notebook:
[Process Glasses Dataset](https://colab.research.google.com/github/amin8452/Pfa-3D/blob/master/notebooks/process_glasses_dataset.ipynb)
