"""
Visualize Filtering Stages for TypeE
------------------------------------

This script generates overlay images showing branches at different filtering stages:
- Before filter #6 (initial branches)
- Before filter #7 (after distance filter)
- Before filter #8 (after removing overlaps)
- After filter #8 (final, after selecting top N)

For each image in TypeE, it creates overlay images of skeleton branches on the original image.
"""

import os
import cv2
import math
import numpy as np
from glob import glob
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Functions from 3distance_map_csv.py
# ----------------------------------------------------------------------
def apply_skeletonization(image):
    """
    Convert a binary 0/255 mask to 0/1, apply skeletonization,
    and convert the result back to 0/255.
    """
    binary = image // 255
    skeleton = skeletonize(binary)
    return (skeleton * 255).astype(np.uint8)


def get_nearest_skeleton_to_center(skeleton_image):
    """
    Identify the skeleton pixel closest to the geometric image center.
    """
    h, w = skeleton_image.shape
    cy, cx = h // 2, w // 2
    pixels = np.argwhere(skeleton_image == 255)
    if len(pixels) == 0:
        return None
    dist = np.linalg.norm(pixels - np.array([cy, cx]), axis=1)
    return tuple(pixels[np.argmin(dist)])


def get_neighbors(skel_img, node):
    """
    Return all 8-connected neighboring skeleton pixels (value = 255).
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


def dfs_build_graph_no_filter(skel_img, node, parent, graph, visited):
    """
    DFS traversal starting from the center node. Construct a graph where each
    skeleton pixel is represented as a node connected to its 8-neighbors.
    """
    visited.add(node)
    graph[node] = []
    nbrs = get_neighbors(skel_img, node)
    if parent is not None:
        nbrs = [n for n in nbrs if n != parent]

    for n in nbrs:
        graph[node].append(n)
        graph.setdefault(n, [])
        if node not in graph[n]:
            graph[n].append(node)
        if n not in visited:
            dfs_build_graph_no_filter(skel_img, n, node, graph, visited)


def find_endpoints_in_graph(graph, center_node):
    """
    Identify branch endpoints: nodes with no unvisited children during DFS.
    Returns:
        branches: {endpoint: [list of skeleton nodes from center to endpoint]}
        endpoints: list of endpoints
    """
    branches = {}
    endpoints = []
    visited = set()

    def dfs(u, path, parent):
        visited.add(u)
        nbrs = [v for v in graph.get(u, []) if v != parent and v not in visited]
        if not nbrs and u != center_node:
            branches[u] = path
            endpoints.append(u)
        for v in nbrs:
            dfs(v, path + [v], u)

    dfs(center_node, [center_node], None)
    return branches, endpoints


def filter_branches_by_distance(branches, center_node, image_width, ratio):
    """
    Keep branches whose endpoint distance from the center exceeds image_width * ratio.
    """
    thr = image_width * ratio
    out = {}
    eps = []
    for ep, path in branches.items():
        dy = ep[0] - center_node[0]
        dx = ep[1] - center_node[1]
        dist = math.sqrt(dy*dy + dx*dx)
        if dist >= thr:
            out[ep] = path
            eps.append(ep)
    return out, eps


def remove_short_overlapping_branches(filtered_branches, common_len_thr):
    """
    Remove shorter branches when two branches share a long common initial segment.
    """
    fb = dict(filtered_branches)
    eps = list(fb.keys())
    remove = set()

    for i in range(len(eps)):
        ep1 = eps[i]
        path1 = fb[ep1]
        for j in range(i + 1, len(eps)):
            ep2 = eps[j]
            path2 = fb[ep2]
            common = 0
            for k in range(min(len(path1), len(path2))):
                if path1[k] == path2[k]:
                    common += 1
                else:
                    break
            if common >= common_len_thr:
                if len(path1) < len(path2):
                    remove.add(ep1)
                else:
                    remove.add(ep2)

    for r in remove:
        fb.pop(r, None)
    return fb, list(fb.keys())


def select_top_n_longest_branches(branches, n=3):
    """
    Select the longest N branches based on path length.
    """
    if len(branches) <= n:
        return branches, list(branches.keys())
    items = sorted(branches.items(), key=lambda kv: len(kv[1]), reverse=True)
    items = items[:n]
    out = dict(items)
    return out, list(out.keys())


def draw_branches_on_image(image, branches, skeleton, center_node, title_suffix=""):
    """
    Draw branches on the image and create an overlay visualization.
    
    Args:
        image: Original grayscale image
        branches: Dictionary of {endpoint: [list of nodes]}
        skeleton: Skeleton image
        center_node: Center node coordinates
        title_suffix: Additional text for the title
    
    Returns:
        overlay_image: RGB image with branches drawn
    """
    # Convert to RGB for color overlay
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Draw full skeleton in light gray
    skeleton_mask = skeleton > 0
    overlay[skeleton_mask] = [200, 200, 200]  # Light gray
    
    # Draw center node (BGR format: blue)
    if center_node:
        cy, cx = center_node
        cv2.circle(overlay, (cx, cy), 5, (255, 0, 0), -1)  # Blue center in BGR
    
        # Draw each branch in a different color (BGR format for OpenCV)
        colors_bgr = [
            (0, 0, 255),      # Red in BGR
            (0, 255, 0),      # Green in BGR
            (255, 0, 0),      # Blue in BGR
            (0, 255, 255),    # Yellow in BGR
            (255, 0, 255),    # Magenta in BGR
            (255, 255, 0),    # Cyan in BGR
            (0, 128, 255),    # Orange in BGR
            (255, 0, 128),    # Purple in BGR
        ]
        
        for idx, (endpoint, path) in enumerate(branches.items()):
            color_bgr = colors_bgr[idx % len(colors_bgr)]
            
            # Draw branch path
            for i in range(len(path) - 1):
                y1, x1 = path[i]
                y2, x2 = path[i + 1]
                cv2.line(overlay, (x1, y1), (x2, y2), color_bgr, 2)
            
            # Draw endpoint
            ey, ex = endpoint
            cv2.circle(overlay, (ex, ey), 4, color_bgr, -1)
    
    # Convert BGR to RGB for matplotlib display
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay_rgb


def process_typee_images():
    """
    Process TypeE images and generate overlay visualizations for each filtering stage.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "2dendritic_images", "TypeE")
    output_dir = os.path.join(script_dir, "filtering_stages_visualization", "TypeE")
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = sorted(glob(os.path.join(input_dir, "*.png")))
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Processing {len(image_paths)} images from TypeE...")
    
    for img_path in image_paths:
        fname = os.path.basename(img_path)
        base_name = os.path.splitext(fname)[0]
        
        print(f"  Processing: {fname}")
        
        # Read original image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"    Failed to read {fname}. Skipping.")
            continue
        
        # Skeletonization
        skeleton = apply_skeletonization(image)
        center_node = get_nearest_skeleton_to_center(skeleton)
        if center_node is None:
            print(f"    No center node found in {fname}. Skipping.")
            continue
        
        # Build graph
        graph = {}
        visited = set()
        dfs_build_graph_no_filter(skeleton, center_node, None, graph, visited)
        branches, endpoints = find_endpoints_in_graph(graph, center_node)
        
        # Stage 1: Before filter #6 (initial branches)
        branches_before_6 = dict(branches)
        overlay_before_6 = draw_branches_on_image(
            image, branches_before_6, skeleton, center_node, "Before Filter #6"
        )
        
        h, w = skeleton.shape
        
        # Apply filter #6
        branches_after_6, _ = filter_branches_by_distance(branches, center_node, w, 0.04)
        
        # Stage 2: Before filter #7 (after filter #6)
        branches_before_7 = dict(branches_after_6)
        overlay_before_7 = draw_branches_on_image(
            image, branches_before_7, skeleton, center_node, "Before Filter #7"
        )
        
        # Apply filter #7
        branches_after_7, _ = remove_short_overlapping_branches(branches_after_6, 50)
        
        # Stage 3: Before filter #8 (after filter #7)
        branches_before_8 = dict(branches_after_7)
        overlay_before_8 = draw_branches_on_image(
            image, branches_before_8, skeleton, center_node, "Before Filter #8"
        )
        
        # Apply filter #8
        branches_after_8, _ = select_top_n_longest_branches(branches_after_7, 3)
        
        # Stage 4: After filter #8 (final)
        overlay_after_8 = draw_branches_on_image(
            image, branches_after_8, skeleton, center_node, "After Filter #8"
        )
        
        # Create a figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        fig.suptitle(f'{fname} - Filtering Stages', fontsize=16, fontweight='bold')
        
        # Stage 1: Before filter #6
        axes[0, 0].imshow(overlay_before_6)
        axes[0, 0].set_title(f'Before Filter #6\n(Initial branches: {len(branches_before_6)})', fontsize=12)
        axes[0, 0].axis('off')
        
        # Stage 2: Before filter #7
        axes[0, 1].imshow(overlay_before_7)
        axes[0, 1].set_title(f'Before Filter #7\n(After distance filter: {len(branches_before_7)})', fontsize=12)
        axes[0, 1].axis('off')
        
        # Stage 3: Before filter #8
        axes[1, 0].imshow(overlay_before_8)
        axes[1, 0].set_title(f'Before Filter #8\n(After removing overlaps: {len(branches_before_8)})', fontsize=12)
        axes[1, 0].axis('off')
        
        # Stage 4: After filter #8
        axes[1, 1].imshow(overlay_after_8)
        axes[1, 1].set_title(f'After Filter #8\n(Final (top N): {len(branches_after_8)})', fontsize=12)
        axes[1, 1].axis('off')
        
        # Save the figure
        output_path = os.path.join(output_dir, f"{base_name}_filtering_stages.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {output_path}")
    
    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    process_typee_images()

