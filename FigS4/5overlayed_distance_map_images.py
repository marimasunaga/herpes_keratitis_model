import os
import cv2
import math
import numpy as np
from glob import glob
from skimage.morphology import skeletonize
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import distance_transform_edt


# ----------------------------------------------------------------------
# Skeletonization
# ----------------------------------------------------------------------
def apply_skeletonization(image):
    binary = image // 255
    skeleton = skeletonize(binary)
    return (skeleton * 255).astype(np.uint8)


# ----------------------------------------------------------------------
# Find skeleton pixel closest to image center
# ----------------------------------------------------------------------
def get_nearest_skeleton_to_center(skeleton_image):
    h, w = skeleton_image.shape
    cy, cx = h // 2, w // 2
    pixels = np.argwhere(skeleton_image == 255)
    if len(pixels) == 0:
        return None
    dist = np.linalg.norm(pixels - np.array([cy, cx]), axis=1)
    return tuple(pixels[np.argmin(dist)])


# ----------------------------------------------------------------------
# 8-neighborhood
# ----------------------------------------------------------------------
def get_neighbors(skel_img, node):
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
# DFS-based graph construction
# ----------------------------------------------------------------------
def dfs_build_graph_no_filter(skel_img, node, parent, graph, visited):
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
# Extract branches
# ----------------------------------------------------------------------
def find_endpoints_in_graph(graph, center_node):
    branches = {}
    visited = set()

    def dfs(u, path, parent):
        visited.add(u)
        nbrs = [v for v in graph.get(u, []) if v != parent and v not in visited]
        if not nbrs and u != center_node:
            branches[u] = path
        for v in nbrs:
            dfs(v, path + [v], u)

    dfs(center_node, [center_node], None)
    return branches


# ----------------------------------------------------------------------
# Distance-based filtering
# ----------------------------------------------------------------------
def filter_branches_by_distance(branches, center_node, image_width, ratio):
    thr = image_width * ratio
    out = {}
    for ep, path in branches.items():
        dy = ep[0] - center_node[0]
        dx = ep[1] - center_node[1]
        if math.sqrt(dy*dy + dx*dx) >= thr:
            out[ep] = path
    return out


# ----------------------------------------------------------------------
# Remove overlapping short branches
# ----------------------------------------------------------------------
def remove_short_overlapping_branches(branches, common_len_thr):
    fb = dict(branches)
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
                remove.add(ep1 if len(path1) < len(path2) else ep2)

    for r in remove:
        fb.pop(r, None)
    return fb


# ----------------------------------------------------------------------
# Select top-N longest branches
# ----------------------------------------------------------------------
def select_top_n_longest_branches(branches, n=3):
    if len(branches) <= n:
        return branches
    items = sorted(branches.items(), key=lambda kv: len(kv[1]), reverse=True)
    return dict(items[:n])


# ======================================================================
# Overlay visualization
# ======================================================================
def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(glob(os.path.join(input_dir, "*.png")))

    sigma = 5

    for in_path in image_paths:
        
        fname = os.path.basename(in_path)
        out_path = os.path.join(output_dir, fname)

        image = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        dist_map = distance_transform_edt(image > 0)

        color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        skeleton = apply_skeletonization(image)

        center = get_nearest_skeleton_to_center(skeleton)
        if center is None:
            continue

        graph = {}
        visited = set()
        dfs_build_graph_no_filter(skeleton, center, None, graph, visited)

        branches = find_endpoints_in_graph(graph, center)

        h, w = skeleton.shape
        branches = filter_branches_by_distance(branches, center, w, 0.04)
        branches = remove_short_overlapping_branches(branches, 50)
        branches = select_top_n_longest_branches(branches, 3)

        overlay = color_img.copy()

        for path in branches.values():
            trimmed_path = list(path)

            while len(trimmed_path) > 0:
                y, x = trimmed_path[-1]   # distal (tip) side
                if dist_map[y, x] < 3:
                    trimmed_path.pop()
                else:
                    break

            # Skip if too short after trimming
            if len(trimmed_path) < 25:
                continue

            path = trimmed_path
            L = len(path)

            # --- distance-map equivalent root detection ---
            distances = np.arange(L)
            smoothed = gaussian_filter1d(distances, sigma=sigma)

            d1 = np.gradient(smoothed)
            root_idx = None
            for i in range(21, len(d1)):
                if d1[i-1] < 0 and d1[i] > 0:
                    root_idx = i
                    break
            if root_idx is None:
                root_idx = int(np.argmin(smoothed[20:]) + 20)

            root_left  = max(0, root_idx - 5)
            root_right = min(L, root_idx + 6)

            tip_start = int(L * 0.8)
            tip_end   = int(L * 0.9)

            R = 3   # thickness (circle radius)
            T = -1  # thickness (filled)

            for i, (y, x) in enumerate(path):

                # Root region (light blue)
                if root_left <= i < root_right:
                    cv2.circle(overlay, (x, y), R, (255, 255, 0), T)   # light blue (BGR)

                # Tip region (orange)
                elif tip_start <= i < tip_end:
                    cv2.circle(overlay, (x, y), R, (0, 165, 255), T)   # orange (BGR)

                # Middle region (gray)
                else:
                    cv2.circle(overlay, (x, y), R, (180, 180, 180), T)  # gray

        cv2.imwrite(out_path, overlay)
        print(f"Saved overlay: {out_path}")


# ======================================================================
# Run
# ======================================================================
if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))

    input_root = os.path.join(script_dir, "2dendritic_images")
    output_root = os.path.join(script_dir, "5overlayed_distance_map")

    subdirs = ["TypeA", "TypeB", "TypeC", "TypeD", "TypeE", "modified_model"]

    for sd in subdirs:
        in_dir = os.path.join(input_root, sd)
        out_dir = os.path.join(output_root, sd)
        if os.path.isdir(in_dir):
            process_directory(in_dir, out_dir)
