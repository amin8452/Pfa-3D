"""
Script to create the data directory structure
"""

import os

def create_data_structure():
    """Create the data directory structure"""
    # Define the directory structure
    dirs = [
        'data/train/images',
        'data/train/models',
        'data/val/images',
        'data/val/models',
        'data/test/images',
        'data/test/models'
    ]
    
    # Create directories
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")

if __name__ == '__main__':
    create_data_structure()
    print("Data directory structure created successfully!")
