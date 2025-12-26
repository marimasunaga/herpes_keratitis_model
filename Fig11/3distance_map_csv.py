"""
Skeleton-Based Branch Extraction and Distance-Map Analysis
----------------------------------------------------------

This script extracts dendritic branch structures from binary lesion masks.
For each image, the pipeline performs:

    1. Skeletonization
    2. Detection of the center skeleton node
    3. Graph construction of the skeleton using depth-first search (DFS)
    4. Identification of branch endpoints
    5. Distance-based filtering of short branches
    6. Removal of short overlapping branches
    7. Selection of the top N longest branches
    8. Extraction of distance-transform values along each branch
    9. Export of branch profiles into a CSV file

The script automatically processes the following directory structure:

    script_dir/
        ├── 2dendritic_images/
        │       ├── TypeA/
        │       ├── TypeB/
        │       ├── TypeC/
        │       ├── TypeD/
        │       ├── TypeE/
        │       └── modified_model/
        └── distance_map_csv/   ← CSV output is stored here

Each subdirectory (TypeA–E, modified_model) yields one CSV file containing
branch distance profiles for all images in that directory.

Processing logic strictly follows the user-provided code; only structure and
formatting were optimized for readability and publication.
"""

import os
import cv2
import math
import numpy as np
import pandas as pd
from glob import glob
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt


# ----------------------------------------------------------------------
# 1. Skeletonization
# ----------------------------------------------------------------------
def apply_skeletonization(image):
    """
    Convert a binary 0/255 mask to 0/1, apply skeletonization,
    and convert the result back to 0/255.
    """
    binary = image // 255
    skeleton = skeletonize(binary)
    return (skeleton * 255).astype(np.uint8)


# ----------------------------------------------------------------------
# 2. Find the skeleton pixel closest to the image center
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# 3. 8-connected neighbors
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# 4. Build skeleton graph via DFS
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# 5. Extract branch endpoints from the graph
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# 6. Distance-based branch filtering
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# 7. Remove short overlapping branches
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# 8. Keep only the top-N longest branches
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# 9. Extract distance-map values along each branch
# ----------------------------------------------------------------------
def get_branch_distance_values(branches, dist_map):
    """
    Return distance-map values for each branch.
    """
    out = {}
    for ep, coords in branches.items():
        out[ep] = [dist_map[y, x] for (y, x) in coords]
    return out


# ======================================================================
# Directory-level processing (one CSV per directory)
# ======================================================================
def process_directory(input_dir, output_root):
    """
    Process all PNG images in a directory and produce a single CSV file
    containing all branch distance profiles.
    """
    image_paths = sorted(glob(os.path.join(input_dir, "*.png")))
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    all_branch_data = []

    for img_path in image_paths:
        fname = os.path.basename(img_path)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"{fname}: failed to read. Skipping.")
            continue

        skeleton = apply_skeletonization(image)
        center_node = get_nearest_skeleton_to_center(skeleton)
        if center_node is None:
            print(f"{fname}: no center node found. Skipping.")
            continue

        graph = {}
        visited = set()
        dfs_build_graph_no_filter(skeleton, center_node, None, graph, visited)
        branches, endpoints = find_endpoints_in_graph(graph, center_node)

        h, w = skeleton.shape
        branches, _ = filter_branches_by_distance(branches, center_node, w, 0.04)
        branches, _ = remove_short_overlapping_branches(branches, 50)
        branches, _ = select_top_n_longest_branches(branches, 3)

        dist_map = distance_transform_edt(image)
        branch_values = get_branch_distance_values(branches, dist_map)

        for ep, vals in branch_values.items():
            bid = f"{fname}_{ep[1]}_{ep[0]}"
            all_branch_data.append([bid] + vals)

    if not all_branch_data:
        print(f"No branch data found in {input_dir}")
        return

    max_len = max(len(row) for row in all_branch_data) - 1
    columns = ["branch_id"] + [f"node_{i}" for i in range(max_len)]
    df = pd.DataFrame(all_branch_data, columns=columns)

    csv_path = os.path.join(output_root, f"{os.path.basename(input_dir)}_branches.csv")
    df.to_csv(csv_path, index=False)

    print(f"Saved CSV: {csv_path}")


# ======================================================================
# Execute for TypeA–E and modified_model
# ======================================================================
if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "2dendritic_images")

    output_root = os.path.join(script_dir, "3distance_map_csv")
    os.makedirs(output_root, exist_ok=True)

    subdirs = ["TypeA", "TypeB", "TypeC", "TypeD", "TypeE", "modified_model"]

    for sd in subdirs:
        d = os.path.join(base_dir, sd)
        if os.path.isdir(d):
            process_directory(d, output_root)
        else:
            print(f"Directory not found: {d}")
