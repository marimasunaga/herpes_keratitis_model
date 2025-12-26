"""
Clinical Image Processing: Skeletonization and Distance Map CSV Generation
---------------------------------------------------------------------------

This script processes clinical images (cropped images) and performs:
    1. Skeletonization of binary images
    2. Interactive selection of start and goal positions on the skeleton
    3. Path finding between start and goal nodes using breadth-first search (BFS)
    4. Extraction of distance transform values along the path
    5. Output to CSV file

The distance transform values represent the distance from each skeleton pixel
to the nearest background pixel, providing a measure of lesion width.
"""

import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt


# ----------------------------------------------------------------------
# Image Binarization
# ----------------------------------------------------------------------
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
    # Check if image is already binary (contains only 0 and 255)
    unique_vals = np.unique(image)
    if len(unique_vals) <= 2 and (0 in unique_vals or 255 in unique_vals):
        if 0 in unique_vals and 255 in unique_vals:
            return image
    
    # Apply thresholding
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary


# ----------------------------------------------------------------------
# Skeletonization
# ----------------------------------------------------------------------
def apply_skeletonization(image):
    """
    Apply skeletonization to a binary image.
    Converts binary image (0/255) to (0/1), applies skeletonization,
    and converts back to (0/255).
    
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


# ----------------------------------------------------------------------
# Find Nearest Skeleton Node
# ----------------------------------------------------------------------
def get_nearest_skeleton_node(skeleton_image, click_point):
    """
    Find the nearest skeleton node to the clicked point.
    
    Parameters:
    -----------
    skeleton_image : numpy.ndarray
        Binary skeleton image (0 or 255)
    click_point : tuple
        Clicked point in OpenCV coordinate system (x, y)
    
    Returns:
    --------
    tuple or None
        Nearest skeleton node coordinates (y, x) in numpy array format,
        or None if no skeleton pixels are found
    """
    pixels = np.argwhere(skeleton_image == 255)
    if len(pixels) == 0:
        return None
    
    # Convert OpenCV coordinates (x, y) to numpy array format (y, x)
    click_y, click_x = click_point[1], click_point[0]
    dist = np.linalg.norm(pixels - np.array([click_y, click_x]), axis=1)
    return tuple(pixels[np.argmin(dist)])


# ----------------------------------------------------------------------
# Interactive Start/Goal Selection
# ----------------------------------------------------------------------
def select_start_goal_interactive(image, skeleton, image_name):
    """
    Display image with overlaid skeleton and allow interactive selection
    of start and goal positions by clicking.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input grayscale or color image
    skeleton : numpy.ndarray
        Binary skeleton image (0 or 255)
    image_name : str
        Name of the image file for display
    
    Returns:
    --------
    tuple
        (start_node, goal_node) where each is (y, x) coordinates of skeleton nodes,
        or (None, None) if selection is cancelled or skipped
    """
    # Convert image to color if grayscale
    if len(image.shape) == 2:
        color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        color_img = image.copy()
    
    # Overlay skeleton in green
    overlay = color_img.copy()
    skeleton_colored = np.zeros_like(color_img)
    skeleton_colored[skeleton == 255] = [0, 255, 0]  # Green color (BGR)
    overlay = cv2.addWeighted(overlay, 0.7, skeleton_colored, 0.3, 0)
    
    # Variables to store clicked positions
    start_point = [None]
    goal_point = [None]
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Find nearest skeleton node to click position
            nearest_node = get_nearest_skeleton_node(skeleton, (x, y))
            if nearest_node:
                node_y, node_x = nearest_node
                # Select start and goal sequentially
                if start_point[0] is None:
                    # Select as start position
                    start_point[0] = (node_x, node_y)
                    print(f"  Start position selected: ({node_y}, {node_x})")
                elif goal_point[0] is None:
                    # Select as goal position
                    goal_point[0] = (node_x, node_y)
                    print(f"  Goal position selected: ({node_y}, {node_x})")
                else:
                    # Reset start and reselect if both are already selected
                    start_point[0] = (node_x, node_y)
                    goal_point[0] = None
                    print(f"  Start position reselected: ({node_y}, {node_x})")
                
                # Draw markers for selected points
                display_overlay = overlay.copy()
                if start_point[0]:
                    sx, sy = start_point[0]
                    cv2.circle(display_overlay, (sx, sy), 8, (255, 0, 0), -1)  # Blue: Start
                    cv2.putText(display_overlay, 'START', (sx+10, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                if goal_point[0]:
                    gx, gy = goal_point[0]
                    cv2.circle(display_overlay, (gx, gy), 8, (0, 0, 255), -1)  # Red: Goal
                    cv2.putText(display_overlay, 'GOAL', (gx+10, gy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow(window_name, display_overlay)
    
    # Display window
    window_name = f'Select Start & Goal: {image_name}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, overlay)
    
    print(f"\n{image_name}:")
    print("  Step 1: Click on the image to select Start position (blue marker)")
    print("  Step 2: Click on the image to select Goal position (red marker)")
    print("  Step 3: Press 'Enter' to confirm, 'r' to reset, 'q' to skip, 'Esc' to cancel")
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13 or key == 10:  # Enter key
            if start_point[0] is not None and goal_point[0] is not None:
                start_node = get_nearest_skeleton_node(skeleton, start_point[0])
                goal_node = get_nearest_skeleton_node(skeleton, goal_point[0])
                cv2.destroyAllWindows()
                if start_node and goal_node:
                    print(f"  Confirmed: Start={start_node}, Goal={goal_node}")
                    return start_node, goal_node
                else:
                    print("  Error: Start or Goal position not found.")
            else:
                print("  Error: Please select both Start and Goal positions.")
        elif key == ord('r') or key == ord('R'):  # Reset
            start_point[0] = None
            goal_point[0] = None
            cv2.imshow(window_name, overlay)
            print("  Reset.")
        elif key == ord('q') or key == ord('Q'):
            cv2.destroyAllWindows()
            print("  Skipped.")
            return None, None
        elif key == 27:  # Esc key
            cv2.destroyAllWindows()
            print("  Cancelled.")
            return None, None


# ----------------------------------------------------------------------
# Path Finding Between Nodes (BFS)
# ----------------------------------------------------------------------
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
    
    # BFS path search
    queue = [(start_node, [start_node])]  # (current_node, path)
    visited = {start_node}
    
    while queue:
        current, path = queue.pop(0)
        
        # Get neighboring nodes
        neighbors = get_neighbors(skeleton, current)
        
        for neighbor in neighbors:
            if neighbor == goal_node:
                # Goal reached
                return path + [neighbor]
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    # Path not found
    return None


# ----------------------------------------------------------------------
# Get 8-Connected Neighbors
# ----------------------------------------------------------------------
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








# ----------------------------------------------------------------------
# Extract Distance Map Values Along Path
# ----------------------------------------------------------------------
def get_path_distance_values(path, dist_map):
    """
    Extract distance transform values along the specified path.
    
    Parameters:
    -----------
    path : list
        List of nodes along the path [(y, x), ...]
    dist_map : numpy.ndarray
        Distance transform map (Euclidean distance to nearest background pixel)
    
    Returns:
    --------
    list
        List of distance values along the path
    """
    if path is None:
        return []
    return [dist_map[y, x] for (y, x) in path]


# ======================================================================
# Directory-Level Processing (One CSV per Directory)
# ======================================================================
def process_directory(input_dir, output_csv, interactive=True):
    """
    Process all PNG images in a directory and generate a single CSV file
    containing distance transform profiles for all paths.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing input PNG images
    output_csv : str
        Path to output CSV file
    interactive : bool
        If True, interactively select Start/Goal positions for each image
    """
    image_paths = sorted(glob(os.path.join(input_dir, "*.png")))
    if not image_paths:
        print(f"No images found in: {input_dir}")
        return

    all_branch_data = []

    for img_path in image_paths:
        fname = os.path.basename(img_path)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"{fname}: Failed to load image. Skipping.")
            continue

        # Binarize image
        binary_image = binarize_image(image, threshold=127)
        
        # Apply skeletonization
        skeleton = apply_skeletonization(binary_image)
        
        # Determine Start and Goal positions
        start_node = None
        goal_node = None
        
        # Interactive selection
        if interactive:
            start_node, goal_node = select_start_goal_interactive(image, skeleton, fname)
        else:
            print(f"{fname}: Please run in interactive mode.")
            continue
        
        if start_node is None or goal_node is None:
            print(f"{fname}: Start/Goal positions not set. Skipping.")
            continue

        # Find path between Start and Goal
        path = find_path_between_nodes(skeleton, start_node, goal_node)
        if path is None:
            print(f"{fname}: No path found between Start and Goal. Skipping.")
            continue
        
        print(f"  Path found: {len(path)} nodes")

        # Calculate distance transform map
        dist_map = distance_transform_edt(binary_image)
        
        # Extract distance map values along the path
        path_distance_values = get_path_distance_values(path, dist_map)
        
        if path_distance_values:
            # Generate path ID (includes start and goal coordinates)
            path_id = f"{fname}_start{start_node[1]}_{start_node[0]}_goal{goal_node[1]}_{goal_node[0]}"
            all_branch_data.append([path_id] + path_distance_values)

    if not all_branch_data:
        print(f"No path data found in: {input_dir}")
        return

    # Create DataFrame with appropriate columns
    max_len = max(len(row) for row in all_branch_data) - 1
    columns = ["path_id"] + [f"node_{i}" for i in range(max_len)]
    df = pd.DataFrame(all_branch_data, columns=columns)

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"CSV saved to: {output_csv}")


# ======================================================================
# Main Execution
# ======================================================================
if __name__ == "__main__":

    import sys
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = script_dir  # Process images in the same directory as this script
    output_csv = os.path.join(script_dir, "clinical_branches.csv")
    
    # Run in interactive mode (manually select Start/Goal positions)
    interactive = True
    print("Running in interactive mode.")
    print("  Click on each image to select Start and Goal positions.")
    print("  Selected Start/Goal positions will be saved in the CSV file's path_id column.")

    process_directory(input_dir, output_csv, interactive=interactive)

