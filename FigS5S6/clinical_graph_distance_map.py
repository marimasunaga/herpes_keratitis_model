"""
Clinical Image Processing: Distance Map Graph Generation (Root/Tip Value Extraction and Visualization)
------------------------------------------------------------------------------------------------------

Read data from CSV file and generate graphs of distance map values for each path.
Marks and visualizes Root (first local minimum) and Tip (last 10-20%) regions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def detect_first_minimum(smoothed):
    """
    Detect the first local minimum in a smoothed curve.
    A local minimum is a point where the derivative changes from negative to positive.
    
    Parameters:
    -----------
    smoothed : numpy.ndarray
        Smoothed distance values
    
    Returns:
    --------
    int
        Index of the first local minimum
    """
    d1 = np.gradient(smoothed)

    for i in range(1, len(d1)):
        if d1[i-1] < 0 and d1[i] > 0:
            return i

    # Fallback: global minimum
    return int(np.argmin(smoothed))


def get_tip_value(distance_values, tip_start_ratio=0.8, tip_end_ratio=0.9):
    """
    Calculate the mean value of the tip region (last 10-20% of the path).
    
    Parameters:
    -----------
    distance_values : numpy.ndarray
        Distance transform values along the path
    tip_start_ratio : float
        Start position ratio for tip region (default: 0.8)
    tip_end_ratio : float
        End position ratio for tip region (default: 0.9)
    
    Returns:
    --------
    float
        Mean distance value in the tip region
    """
    L = len(distance_values)
    tip_start = int(L * tip_start_ratio)
    tip_end = int(L * tip_end_ratio)
    if tip_start < tip_end:
        return np.mean(distance_values[tip_start:tip_end])
    return distance_values[-1] if len(distance_values) > 0 else 0


if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "clinical_branches.csv")
    output_dir = os.path.join(script_dir, "graphs")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        print("Please run clinical_distance_map_csv.py first.")
        exit(1)

    # Read CSV file
    df = pd.read_csv(csv_path)

    # Get node columns (node_0, node_1, ...)
    node_cols = [col for col in df.columns if col.startswith("node_")]
    df[node_cols] = df[node_cols].apply(pd.to_numeric, errors="coerce")

    print(f"Processing: {csv_path}")

    root_tip_data = []

    for path_idx, row in df.iterrows():

        distance_values = row[node_cols].dropna().astype(float).values

        if len(distance_values) == 0:
            continue

        x = np.arange(len(distance_values))

        # Apply Gaussian smoothing
        sigma = 5
        smoothed = gaussian_filter1d(distance_values, sigma=sigma)

        root_idx = detect_first_minimum(smoothed)

        # Calculate root value as mean of ±5 nodes around the detected minimum
        left = max(0, root_idx - 5)
        right = min(len(smoothed), root_idx + 5)
        root_value = np.mean(smoothed[left:right])

        tip_value = get_tip_value(distance_values)

        path_id = row['path_id'] if 'path_id' in row else f"path_{path_idx}"
        root_tip_data.append({
            'path_id': path_id,
            'root_value': root_value,
            'tip_value': tip_value,
            'root_index': root_idx
        })

        plt.figure(figsize=(6, 4))

        plt.plot(x, distance_values, "o-", color="black", markersize=3, alpha=0.4, label="Original data")
        plt.plot(x, smoothed, "-", color="blue", linewidth=2, label=f"Gaussian smoothing (σ={sigma})")
        plt.scatter(root_idx, smoothed[root_idx], color="blue", s=30, label="First local minimum")

        root_left = max(0, root_idx - 5)
        root_right = min(len(distance_values), root_idx + 6)
        plt.axvspan(root_left, root_right, color="lightblue", alpha=0.35, label="Root region (min ±5 nodes)")
        
        tip_start = int(len(distance_values) * 0.8)
        tip_end = int(len(distance_values) * 0.9)
        plt.axvspan(tip_start, tip_end, color="orange", alpha=0.3, label="Tip region (last 10-20%)")

        plt.title(f"Path: {path_id}\nRoot value: {root_value:.2f}, Tip value: {tip_value:.2f}")

        plt.xlabel("Node index along path")
        plt.ylabel("Distance map value")

        plt.grid(True)
        plt.legend()

        safe_path_id = path_id.replace('/', '_').replace('\\', '_').replace(':', '_')
        out_path = os.path.join(output_dir, f"path_{safe_path_id}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"  Saved → {out_path}")

    if root_tip_data:
        root_tip_df = pd.DataFrame(root_tip_data)
        root_tip_csv = os.path.join(script_dir, "clinical_root_tip_values.csv")
        root_tip_df.to_csv(root_tip_csv, index=False)
        print(f"\nRoot/Tip values saved to: {root_tip_csv}")

    print("\nAll graphs generated.")

