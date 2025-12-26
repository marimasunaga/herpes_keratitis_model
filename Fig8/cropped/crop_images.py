#!/usr/bin/env python3
"""
Center-aligned image cropping script for reducing image dimensions to 30% of original size.
Processes PNG files in subdirectories of 1alpha_shape and raw_grid_images.
"""

import os
from pathlib import Path
from PIL import Image

# Get parent directory of the script directory
script_dir = Path(__file__).parent
parent_dir = script_dir.parent

# Target directories to process
target_dirs = ['1alpha_shape', 'raw_grid_images']

# Crop ratio (30% = 0.3)
crop_ratio = 0.3

def crop_image_center(image_path, output_path, ratio=0.3):
    """
    Crop image to specified ratio while maintaining center alignment.
    
    Args:
        image_path: Path to input image
        output_path: Path to output image
        ratio: Crop ratio
    """
    # Load image
    img = Image.open(image_path)
    width, height = img.size
    
    # Calculate new dimensions 
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    
    # Calculate center-aligned crop coordinates
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    
    # Perform center-aligned cropping
    cropped_img = img.crop((left, top, right, bottom))
    
    # Create output directory if it does not exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save cropped image
    cropped_img.save(output_path)
    print(f"Processed: {image_path} -> {output_path}")

def process_directory(base_dir, target_dir_name, output_base_dir):
    """
    Process all PNG files in the specified directory and its subdirectories.
    
    Args:
        base_dir: Base directory path (input)
        target_dir_name: Name of target directory to process
        output_base_dir: Base directory for output (script directory)
    """
    target_dir = base_dir / target_dir_name
    
    if not target_dir.exists():
        print(f"Warning: Directory {target_dir} does not exist. Skipping...")
        return
    
    # Recursively search for all PNG files in subdirectories
    png_files = list(target_dir.rglob('*.png'))
    
    if not png_files:
        print(f"No PNG files found in {target_dir}")
        return
    
    print(f"\nProcessing {len(png_files)} PNG files in {target_dir}...")
    
    for png_file in png_files:
        # Generate output path maintaining relative directory structure in script directory
        relative_path = png_file.relative_to(base_dir)
        output_path = output_base_dir / relative_path
        
        # Crop image and save to script directory
        crop_image_center(png_file, output_path, crop_ratio)

def main():
    """Main processing function."""
    print(f"Input base directory: {parent_dir}")
    print(f"Output base directory: {script_dir}")
    print(f"Crop ratio: {crop_ratio * 100}%")
    
    for target_dir_name in target_dirs:
        print(f"\n{'='*60}")
        print(f"Processing directory: {target_dir_name}")
        print(f"{'='*60}")
        process_directory(parent_dir, target_dir_name, script_dir)
    
    print(f"\n{'='*60}")
    print("All processing completed!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
