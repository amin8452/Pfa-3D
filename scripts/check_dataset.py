"""
Script to check the dataset directory and list images
"""

import os
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Check the dataset directory')
    parser.add_argument('--dataset_path', type=str, default='data/train/images',
                        help='Path to the dataset of images')
    return parser.parse_args()


def check_directory(path):
    """
    Check the directory and list its contents
    """
    # Convert to absolute path
    abs_path = os.path.abspath(path)
    print(f"Checking directory: {abs_path}")
    
    # Check if the directory exists
    if not os.path.exists(abs_path):
        print(f"Error: Directory {abs_path} does not exist")
        return
    
    # Check if it's a directory
    if not os.path.isdir(abs_path):
        print(f"Error: {abs_path} is not a directory")
        return
    
    # List all files in the directory
    try:
        all_files = os.listdir(abs_path)
        print(f"Directory contains {len(all_files)} files/directories")
        
        # Count files with extensions
        files_with_ext = [f for f in all_files if '.' in f]
        print(f"Files with extensions: {len(files_with_ext)}")
        
        # Count image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        image_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in image_extensions)]
        print(f"Image files: {len(image_files)}")
        
        # Print examples
        if image_files:
            print(f"Examples of image files: {image_files[:5]}")
            
            # Print full paths to a few images
            for img in image_files[:3]:
                img_path = os.path.join(abs_path, img)
                print(f"Full path: {img_path}")
                print(f"File exists: {os.path.exists(img_path)}")
                print(f"File size: {os.path.getsize(img_path)} bytes")
        
        # Check for subdirectories
        subdirs = [f for f in all_files if os.path.isdir(os.path.join(abs_path, f))]
        print(f"Subdirectories: {len(subdirs)}")
        if subdirs:
            print(f"Examples of subdirectories: {subdirs[:5]}")
            
            # Check the first subdirectory
            if subdirs:
                subdir_path = os.path.join(abs_path, subdirs[0])
                print(f"\nChecking subdirectory: {subdir_path}")
                subdir_files = os.listdir(subdir_path)
                print(f"Subdirectory contains {len(subdir_files)} files")
                if subdir_files:
                    print(f"Examples: {subdir_files[:5]}")
        
    except Exception as e:
        print(f"Error listing directory: {e}")


def main():
    args = parse_args()
    check_directory(args.dataset_path)
    
    # Also check parent directory
    parent_dir = os.path.dirname(args.dataset_path)
    print(f"\nChecking parent directory: {parent_dir}")
    check_directory(parent_dir)
    
    # Check current working directory
    cwd = os.getcwd()
    print(f"\nCurrent working directory: {cwd}")
    
    # List all directories in the project root
    print("\nDirectories in project root:")
    for item in os.listdir(cwd):
        if os.path.isdir(os.path.join(cwd, item)):
            print(f"- {item}")


if __name__ == '__main__':
    main()
