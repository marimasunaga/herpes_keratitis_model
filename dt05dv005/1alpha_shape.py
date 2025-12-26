"""
Alpha-shape based post-processing with solid interior filling
"""

import os
import cv2
import glob
import numpy as np
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union
from scipy.spatial import Delaunay


# --------------------------------------------------------------
# Alpha Shape construction
# --------------------------------------------------------------
def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) for a set of 2D points.
    """

    if len(points) < 4:
        return MultiPoint(points).convex_hull

    tri = Delaunay(points)
    triangles = points[tri.simplices]
    faces = []

    for tri_pts in triangles:
        a, b, c = tri_pts

        # edge lengths
        ab = np.linalg.norm(a - b)
        bc = np.linalg.norm(b - c)
        ca = np.linalg.norm(c - a)

        # triangle area (Heron)
        s = (ab + bc + ca) / 2.0
        area = np.sqrt(max(s * (s - ab) * (s - bc) * (s - ca), 0.0))
        if area == 0:
            continue

        # circumradius
        R = (ab * bc * ca) / (4.0 * area)

        # keep triangles with small circumradius
        if R < 1.0 / alpha:
            faces.append(Polygon(tri_pts))

    if not faces:
        return MultiPoint(points).convex_hull

    return unary_union(faces)


# --------------------------------------------------------------
# Process a single image
# --------------------------------------------------------------
def process_single_image(input_path, output_path, alpha=0.02):

    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Warning: cannot read {input_path}")
        return

    # binary conversion
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # contour extraction
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print(f"Skipping (no contours): {input_path}")
        return

    points = np.vstack([cnt[:, 0, :] for cnt in contours])
    if len(points) < 3:
        print(f"Skipping (too few points): {input_path}")
        return

    # compute alpha shape
    shape = alpha_shape(points, alpha)
    shape = shape.buffer(0)  # fix geometry


    result = np.zeros_like(binary)
    result[binary == 255] = 255  # keep all original white pixels

    if shape.geom_type == "Polygon":
        exterior = np.array(shape.exterior.coords, np.int32)
        cv2.fillPoly(result, [exterior], 255)

    elif shape.geom_type == "MultiPolygon":
        # fill all exteriors (usually only one large polygon)
        for geom in shape.geoms:
            exterior = np.array(geom.exterior.coords, np.int32)
            cv2.fillPoly(result, [exterior], 255)

    # ----------------------------------------------------------
    # Hole filling using contour hierarchy (optional but robust)
    # ----------------------------------------------------------
    contours2, hierarchy = cv2.findContours(result, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        for i in range(len(contours2)):
            # a contour with a parent → hole → fill with white
            if hierarchy[0][i][3] != -1:
                cv2.drawContours(result, [contours2[i]], -1, 255, -1)

    # save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, result)
    print(f"Processed: {output_path}")


# --------------------------------------------------------------
# Recursively process a directory
# --------------------------------------------------------------
def process_directory_recursive(input_root, output_root, alpha=0.02):

    image_paths = glob.glob(os.path.join(input_root, "**/*.*"), recursive=True)

    for path in image_paths:
        if not path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
            continue

        rel = os.path.relpath(path, input_root)
        out_path = os.path.join(output_root, rel)

        process_single_image(path, out_path, alpha)


# --------------------------------------------------------------
# main execution
# --------------------------------------------------------------
if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "0round_cell_images")
    output_dir = os.path.join(script_dir, "1alpha_shape")

    alpha_value = 0.3
    process_directory_recursive(input_dir, output_dir, alpha=alpha_value)
