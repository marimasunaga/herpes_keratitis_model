import os
from collections import defaultdict
import numpy as np
import umap
import matplotlib.pyplot as plt

def perform_umap(embeddings, seed, n_components=2):
    reducer = umap.UMAP(n_components=n_components, random_state=seed)
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings

def plot_umap_by_model(reduced_embeddings, image_paths, output_path="RN50_UMAP_colored_filtered_10_20.png"):
    fig, ax = plt.subplots(figsize=(8, 5))

    legend_order = ["TypeA", "TypeB", "TypeC", "TypeD", "TypeE", "modified model"]

    color_map = {
        "TypeA": "#1f77b4",
        "TypeB": "#2ca02c",
        "TypeC": "#17becf",
        "TypeD": "#9467bd",
        "TypeE": "gold",
        "modified model": "#d62728"
    }

    labeled_points = defaultdict(list)

    for coords, path in zip(reduced_embeddings, image_paths):
        path_parts = path.split(os.sep)
        if "TypeA" in path_parts:
            label = "TypeA"
        elif "TypeB" in path_parts:
            label = "TypeB"
        elif "TypeC" in path_parts:
            label = "TypeC"
        elif "TypeD" in path_parts:
            label = "TypeD"
        elif "TypeE" in path_parts:
            label = "TypeE"
        else:
            label = "modified model"
        
        labeled_points[label].append(coords)

    for label in legend_order:
        if label in labeled_points:
            points = np.array(labeled_points[label])
            ax.scatter(points[:, 0], points[:, 1], label=label, color=color_map[label], alpha=0.8, s=30)

    ax.set_title("UMAP of CLIP Image Embeddings", fontsize=20)
    ax.set_xlabel("UMAP1", fontsize=16)
    ax.set_ylabel("UMAP2", fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.legend(title="Model", loc="best")
    plt.tight_layout()
    plt.savefig(output_path, format='png', dpi=300)

seed = 42
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = script_dir

min_distance = 10
max_distance = 20

model_types = ["TypeA", "TypeB", "TypeC", "TypeD", "TypeE", "modified_model"]
image_embeddings = []
image_paths = []

for model_type in model_types:
    features_file = os.path.join(script_dir, f"RN50-clip_features_{model_type}_filtered_{int(min_distance)}_{int(max_distance)}.npz")
    if os.path.exists(features_file):
        print(f"Loading features from {features_file}...")
        data = np.load(features_file, allow_pickle=True)
        embeddings = data["embeddings"]
        paths = data["paths"].tolist()
        image_embeddings.append(embeddings)
        image_paths.extend(paths)
        print(f"  Loaded {embeddings.shape[0]} images from {model_type}")
    else:
        print(f"Warning: {features_file} not found, skipping {model_type}")

if not image_embeddings:
    raise ValueError(f"No feature files found. Please run CLIP_extract_featuresRN50_filtered.py first.")

image_embeddings = np.vstack(image_embeddings)
print(f"\nTotal image embeddings shape: {image_embeddings.shape}")
print(f"Total number of images: {len(image_paths)}")

print("Performing UMAP dimensionality reduction...")
reduced_embeddings = perform_umap(image_embeddings, seed)

output_path = os.path.join(output_dir, f"RN50_UMAP_colored_filtered_{int(min_distance)}_{int(max_distance)}.png")
plot_umap_by_model(reduced_embeddings, image_paths, output_path=output_path)
print(f"UMAP plot saved to {output_path}")

