"""
Clinical Image Processing: Distance Map Overlay Image Generation
----------------------------------------------------------------

Overlay skeleton and distance map on the original image,
and visualize Root, Tip, and Middle regions with color coding.
"""

import os
import cv2
import numpy as np
import pandas as pd
import re
from glob import glob
from skimage.morphology import skeletonize
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import distance_transform_edt


def binarize_image(image, threshold=127):
    """
    Convert grayscale image to binary image.
    If the image is already binary, return it as is.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input grayscale image
    threshold : int
        Threshold value for binarization (default: 127)
    
    Returns:
    --------
    numpy.ndarray
        Binary image (0 or 255)
    """
    unique_vals = np.unique(image)
    if len(unique_vals) <= 2 and (0 in unique_vals or 255 in unique_vals):
        if 0 in unique_vals and 255 in unique_vals:
            return image
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary


def apply_skeletonization(image):
    """
    Apply skeletonization to a binary image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Binary image (0 or 255)
    
    Returns:
    --------
    numpy.ndarray
        Skeletonized image (0 or 255)
    """
    binary = image // 255
    skeleton = skeletonize(binary)
    return (skeleton * 255).astype(np.uint8)


def get_neighbors(skel_img, node):
    """
    Get all 8-connected skeleton pixels (value=255) adjacent to the given node.
    
    Parameters:
    -----------
    skel_img : numpy.ndarray
        Binary skeleton image (0 or 255)
    node : tuple
        Node coordinates (y, x)
    
    Returns:
    --------
    list
        List of neighboring skeleton node coordinates [(y, x), ...]
    """
    y, x = node
    h, w = skel_img.shape
    nbrs = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if skel_img[ny, nx] == 255:
                    nbrs.append((ny, nx))
    return nbrs


def find_path_between_nodes(skeleton, start_node, goal_node):
    """
    Find the shortest path between start and goal nodes on the skeleton
    using breadth-first search (BFS).
    
    Parameters:
    -----------
    skeleton : numpy.ndarray
        Binary skeleton image (0 or 255)
    start_node : tuple
        Starting node coordinates (y, x)
    goal_node : tuple
        Goal node coordinates (y, x)
    
    Returns:
    --------
    list or None
        List of nodes along the path [(y, x), ...], or None if no path is found
    """
    if start_node == goal_node:
        return [start_node]
    
    queue = [(start_node, [start_node])]
    visited = {start_node}
    
    while queue:
        current, path = queue.pop(0)
        neighbors = get_neighbors(skeleton, current)
        
        for neighbor in neighbors:
            if neighbor == goal_node:
                return path + [neighbor]
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None


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
    return int(np.argmin(smoothed))


def extract_start_goal_from_csv(csv_path):
    """
    Extract start/goal positions from path_id in CSV file.
    Example: "image1.png_start312_35_goal240_245" → start: (35, 312), goal: (245, 240)
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV file containing path_id column
    
    Returns:
    --------
    dict
        Dictionary mapping filename to start/goal positions
    """
    start_goal_dict = {}
    
    if not os.path.exists(csv_path):
        return start_goal_dict
    
    try:
        df = pd.read_csv(csv_path)
        if 'path_id' not in df.columns:
            return start_goal_dict
        
        for path_id in df['path_id']:
            match = re.search(r'^(.+?)_start(\d+)_(\d+)_goal(\d+)_(\d+)$', str(path_id))
            if match:
                fname = match.group(1)
                start_x = int(match.group(2))
                start_y = int(match.group(3))
                goal_x = int(match.group(4))
                goal_y = int(match.group(5))
                
                start_goal_dict[fname] = {
                    'start': {'y': start_y, 'x': start_x},
                    'goal': {'y': goal_y, 'x': goal_x}
                }
        
        print(f"Extracted Start/Goal positions from CSV file: {len(start_goal_dict)} images")
    except Exception as e:
        print(f"Warning: Failed to extract Start/Goal positions from CSV file: {e}")
    
    return start_goal_dict


def process_directory(input_dir, output_dir, csv_path=None):
    """
    Process all PNG images in a directory and generate overlay images
    with Root, Tip, and Middle regions color-coded.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing input PNG images
    output_dir : str
        Directory to save output overlay images
    csv_path : str
        Path to CSV file containing Start/Goal positions
    """
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(glob(os.path.join(input_dir, "*.png")))

    # Load Start/Goal positions from CSV file
    center_nodes_dict = {}
    
    if csv_path and os.path.exists(csv_path):
        center_nodes_dict = extract_start_goal_from_csv(csv_path)
    else:
        print(f"Warning: CSV file not found: {csv_path}")
        print("  Please run clinical_distance_map_csv.py first.")

    sigma = 5

    for in_path in image_paths:
        
        fname = os.path.basename(in_path)
        out_path = os.path.join(output_dir, fname)

        image = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"{fname}: Failed to load image. Skipping.")
            continue

        binary_image = binarize_image(image, threshold=127)
        dist_map = distance_transform_edt(binary_image > 0)
        color_img = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        skeleton = apply_skeletonization(binary_image)

        start_node = None
        goal_node = None
        
        if fname in center_nodes_dict:
            node_data = center_nodes_dict[fname]
            if 'start' in node_data and 'goal' in node_data:
                try:
                    start_node = (node_data['start']['y'], node_data['start']['x'])
                    goal_node = (node_data['goal']['y'], node_data['goal']['x'])
                    print(f"{fname}: Start/Goal positions loaded")
                    print(f"  Start: {start_node}, Goal: {goal_node}")
                except (KeyError, TypeError) as e:
                    print(f"{fname}: Failed to load Start/Goal positions: {e}")
                    print(f"  Data: {node_data}")
        else:
            print(f"{fname}: Start/Goal positions not found in configuration file.")
            print(f"  Keys in configuration file: {list(center_nodes_dict.keys())}")
        
        if start_node is None or goal_node is None:
            print(f"{fname}: Start/Goal positions not set. Skipping.")
            continue

        path = find_path_between_nodes(skeleton, start_node, goal_node)
        if path is None:
            print(f"{fname}: No path found between Start and Goal. Skipping.")
            continue

        print(f"{fname}: Path found ({len(path)} nodes)")

        overlay = color_img.copy()
        L = len(path)

        distance_values = [dist_map[y, x] for (y, x) in path]
        smoothed = gaussian_filter1d(distance_values, sigma=sigma)
        root_idx = detect_first_minimum(smoothed)

        root_left = max(0, root_idx - 5)
        root_right = min(L, root_idx + 6)
        tip_start = int(L * 0.8)
        tip_end = int(L * 0.9)

        R = 3
        T = -1

        for i, (y, x) in enumerate(path):
            if root_left <= i < root_right:
                cv2.circle(overlay, (x, y), R, (255, 255, 0), T)  # light blue (BGR)
            elif tip_start <= i < tip_end:
                cv2.circle(overlay, (x, y), R, (0, 165, 255), T)  # orange (BGR)
            else:
                cv2.circle(overlay, (x, y), R, (180, 180, 180), T)  # gray

        sx, sy = start_node[1], start_node[0]
        gx, gy = goal_node[1], goal_node[0]
        cv2.circle(overlay, (sx, sy), 8, (255, 0, 0), -1)  # blue: Start
        cv2.circle(overlay, (gx, gy), 8, (0, 0, 255), -1)  # red: Goal
        cv2.putText(overlay, 'START', (sx+10, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(overlay, 'GOAL', (gx+10, gy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imwrite(out_path, overlay)
        print(f"Overlay image saved: {out_path}")


if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = script_dir
    output_dir = os.path.join(script_dir, "overlayed_images")
    csv_path = os.path.join(script_dir, "clinical_branches.csv")

    process_directory(input_dir, output_dir, csv_path=csv_path)

