#!/usr/bin/env python3

import os
from pathlib import Path
from PIL import Image

# Get parent directory of the script directory
script_dir = Path(__file__).parent
parent_dir = script_dir.parent

# Crop ratio settings
# Default crop ratio for Type files (30% = 0.3)
default_crop_ratio = 0.3

# Crop ratio for modified_model files 
modified_model_crop_ratio = 0.4

def crop_image_center(image_path, output_path, ratio):
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

def process_type_png_files(base_dir, output_base_dir):
    # Recursively search for all PNG files in parent directory
    all_png_files = list(base_dir.rglob('*.png'))
    
    # Filter PNG files that contain "Type" or "modified_model" in their filename
    target_png_files = [png_file for png_file in all_png_files if 'Type' in png_file.name or 'modified_model' in png_file.name]
    
    for png_file in target_png_files:
        # Determine crop ratio based on filename
        if 'modified_model' in png_file.name:
            ratio = modified_model_crop_ratio
            print(f"  Using modified_model ratio ({ratio * 100}%) for: {png_file.name}")
        else:
            ratio = default_crop_ratio
        
        # Generate output path maintaining relative directory structure in script directory
        relative_path = png_file.relative_to(base_dir)
        output_path = output_base_dir / relative_path
        
        # Crop image and save to script directory
        crop_image_center(png_file, output_path, ratio)

def main():
    """Main processing function."""
    print(f"Input base directory: {parent_dir}")
    print(f"Output base directory: {script_dir}")
    
    print(f"\n{'='*60}")
    print(f"{'='*60}")
    process_type_png_files(parent_dir, script_dir)
    
    print(f"\n{'='*60}")
    print("All processing completed!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
