"""
Training script for glasses 3D reconstruction
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import datetime

from model import Hunyuan3DGlasses
from data_loader_glasses import create_data_loaders
from metrics import chamfer_loss, compute_all_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Train Hunyuan3D model for glasses reconstruction')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for fine-tuning')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for checkpoints and logs')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch in progress_bar:
        # Get data
        images = batch['image'].to(device)
        points = batch['points'].to(device)

        # Forward pass
        pred_points = model(images)

        # Compute loss
        loss = chamfer_loss(pred_points, points)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(train_loader)


def validate(model, val_loader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_metrics = {}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            # Get data
            images = batch['image'].to(device)
            points = batch['points'].to(device)

            # Forward pass
            pred_points = model(images)

            # Compute loss
            loss = chamfer_loss(pred_points, points)
            total_loss += loss.item()

            # Compute metrics
            batch_metrics = compute_all_metrics(pred_points, points)

            # Accumulate metrics
            for k, v in batch_metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v)

    # Average metrics
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    avg_metrics['loss'] = total_loss / len(val_loader)

    return avg_metrics


def save_checkpoint(model, optimizer, epoch, metrics, output_dir, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    # Save latest checkpoint
    checkpoint_path = os.path.join(output_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, checkpoint_path)

    # Save best checkpoint if this is the best model
    if is_best:
        best_path = os.path.join(output_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)

    # Save epoch checkpoint
    epoch_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, epoch_path)


def main():
    # Parse arguments
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Set up output directory
    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = os.path.join('checkpoints', f'run_{timestamp}')

    os.makedirs(args.output_dir, exist_ok=True)

    # Set up tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create data loaders
    data_loaders = create_data_loaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        img_size=config['data']['img_size'],
        num_points=config['data']['num_points']
    )

    # Create model
    model = Hunyuan3DGlasses(
        latent_dim=config['model']['latent_dim'],
        num_points=config['data']['num_points']
    )

    # Load checkpoint for fine-tuning if provided
    start_epoch = 0
    if args.checkpoint is not None:
        print(f'Loading checkpoint from {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Set up for fine-tuning
        if config['training']['fine_tune']:
            print('Setting up model for fine-tuning')
            model.fine_tune(freeze_encoder=config['training']['freeze_encoder'])

    model = model.to(device)

    # Create optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Create scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(start_epoch, config['training']['num_epochs']):
        # Train
        train_loss = train_epoch(model, data_loaders['train'], optimizer, device, epoch)

        # Validate
        val_metrics = validate(model, data_loaders['val'], device)
        val_loss = val_metrics['loss']

        # Update scheduler
        scheduler.step(val_loss)

        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        for k, v in val_metrics.items():
            if k != 'loss':
                writer.add_scalar(f'Metrics/{k}', v, epoch)

        # Print metrics
        print(f'Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
        for k, v in val_metrics.items():
            if k != 'loss':
                print(f'  {k} = {v:.6f}')

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        save_checkpoint(model, optimizer, epoch, val_metrics, args.output_dir, is_best)

    # Final evaluation on test set
    print('Evaluating on test set...')
    test_metrics = validate(model, data_loaders['test'], device)
    print('Test metrics:')
    for k, v in test_metrics.items():
        print(f'  {k} = {v:.6f}')

    # Save test metrics
    with open(os.path.join(args.output_dir, 'test_metrics.yaml'), 'w') as f:
        yaml.dump(test_metrics, f)

    print(f'Training complete. Model saved to {args.output_dir}')


if __name__ == '__main__':
    main()
