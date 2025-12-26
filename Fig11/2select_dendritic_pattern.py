"""
Dendritic / Geographic classification
-------------------------------------

This script:
    (1) Loads the processed masks in "1alpha_shape"
    (2) Computes perimeter / area ratio
    (3) Classifies each image as "dendritic" or "geographic"
    (4) Saves the image into:
            2dendritic_images/
            2geographic_images/
        while preserving the original directory tree.

Assumptions:
    - This script is placed in the same directory as "1alpha_shape"
"""

import os
import cv2
import shutil
import numpy as np


# --------------------------------------------------------------
# Classification function for one image
# --------------------------------------------------------------
def classify_image(image_path, area_thresh, ratio_thresh):
    """
    Classify image into dendritic or geographic based on perimeter/area ratio.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area_total = 0
    perimeter_total = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < area_thresh:
            continue
        perimeter = cv2.arcLength(contour, True)

        area_total += area
        perimeter_total += perimeter

    if area_total == 0:
        return None

    ratio = perimeter_total / area_total
    label = "dendritic" if ratio > ratio_thresh else "geographic"
    return label


# --------------------------------------------------------------
# Process directory recursively (preserve structure)
# --------------------------------------------------------------
def classify_directory_recursive(input_root, output_root_dend, output_root_geo,
                                 area_thresh, ratio_thresh):

    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    for root, _, files in os.walk(input_root):
        for fname in files:
            if not fname.lower().endswith(valid_exts):
                continue

            in_path = os.path.join(root, fname)

            # Determine relative path to rebuild directory structure
            rel_path = os.path.relpath(in_path, input_root)

            # Classify the image
            label = classify_image(in_path, area_thresh, ratio_thresh)
            if label is None:
                print(f"Skipped (no valid contour): {rel_path}")
                continue

            # Choose output directory
            if label == "dendritic":
                out_path = os.path.join(output_root_dend, rel_path)
            else:
                out_path = os.path.join(output_root_geo, rel_path)

            # Create subdirectories as needed
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # Copy the file
            shutil.copy2(in_path, out_path)

            print(f"{rel_path} → {label}")


# --------------------------------------------------------------
# main execution
# --------------------------------------------------------------
if __name__ == "__main__":

    # Get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Input directory (alpha-shape output)
    input_root = os.path.join(script_dir, "1alpha_shape")

    # Output directories
    output_dend = os.path.join(script_dir, "2dendritic_images")
    output_geo  = os.path.join(script_dir, "2geographic_images")

    os.makedirs(output_dend, exist_ok=True)
    os.makedirs(output_geo, exist_ok=True)

    # Classification thresholds
    area_thresh = 0
    ratio_thresh = 0.12

    classify_directory_recursive(
        input_root,
        output_dend,
        output_geo,
        area_thresh=area_thresh,
        ratio_thresh=ratio_thresh,
    )

    print("\nClassification completed.\n")
