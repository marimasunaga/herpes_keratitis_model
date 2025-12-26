import os
import pandas as pd
import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm

def extract_image_filename_from_branch_id(branch_id):
    """
    Extract image filename from branch_id
    Example: plot_a0.500_b1.000_g10.000_d0.500_Dv0.050_iter1040_seed9.png_570_741
    -> plot_a0.500_b1.000_g10.000_d0.500_Dv0.050_iter1040_seed9.png
    """
    # Remove the last two underscore-separated numbers
    parts = branch_id.rsplit('_', 2)
    if len(parts) >= 3:
        # Check if the last two parts are numbers
        try:
            int(parts[-1])
            int(parts[-2])
            return '_'.join(parts[:-2])
        except ValueError:
            return branch_id
    return branch_id

def get_filtered_image_paths(csv_dir, data_type, min_distance=10, max_distance=20):
    """
    Get image filenames from CSV file where node_0 value is within the specified range
    """
    csv_file = os.path.join(csv_dir, f"{data_type}_branches.csv")
    
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found")
        return set()
    
    df = pd.read_csv(csv_file)
    
    # Check if node_0 column exists
    if 'node_0' not in df.columns:
        print(f"Warning: node_0 column not found in {csv_file}")
        return set()
    
    # Filter rows where node_0 value is within the specified range
    filtered_df = df[(df['node_0'] >= min_distance) & (df['node_0'] <= max_distance)]
    
    # Extract image filename from branch_id
    image_filenames = set()
    for branch_id in filtered_df['branch_id']:
        img_filename = extract_image_filename_from_branch_id(branch_id)
        image_filenames.add(img_filename)
    
    print(f"  Found {len(image_filenames)} unique images with node_0 distance {min_distance}-{max_distance} in {data_type}")
    
    return image_filenames

def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50", device=device)
    model.eval()
    return model, preprocess, device

def get_image_embeddings(image_folder, model, preprocess, device, filtered_filenames, batch_size=32):
    """
    Process only images that match the filtered image filenames
    """
    image_paths = []
    
    # Collect image paths that match the filtered filenames
    for root, _, files in os.walk(image_folder):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.png')):
                if fname in filtered_filenames:
                    image_paths.append(os.path.join(root, fname))

    image_embeddings = []
    valid_paths = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing Images"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        batch_valid_paths = []

        for path in batch_paths:
            try:
                image = Image.open(path).convert('RGB')
                batch_images.append(preprocess(image))
                batch_valid_paths.append(path)
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
                continue

        if not batch_images:
            continue

        batch_tensor = torch.stack(batch_images).to(device)
        
        with torch.no_grad():
            batch_features = model.encode_image(batch_tensor)
            batch_features = batch_features / batch_features.norm(dim=1, keepdim=True)
            batch_features_np = batch_features.cpu().numpy()
            image_embeddings.append(batch_features_np)
            valid_paths.extend(batch_valid_paths)

    if not image_embeddings:
        return np.array([]), []

    return np.vstack(image_embeddings), valid_paths

script_dir = os.path.dirname(os.path.abspath(__file__))
input_base_dir = os.path.join(os.path.dirname(script_dir), "Fig11", "2dendritic_images")
output_dir = script_dir

# Directory containing CSV files
fig11_dir = os.path.join(os.path.dirname(script_dir), "Fig11")
csv_dir = os.path.join(fig11_dir, "3distance_map_csv")

min_distance = 10
max_distance = 20

print(f"Filtering images with node_0 distance: {min_distance}-{max_distance}")
print(f"Reading CSV files from: {csv_dir}\n")

print("Loading CLIP model...")
model, preprocess, device = load_clip_model()

batch_size = 64 if device == "cuda" else 16
print(f"Using batch size: {batch_size}\n")

model_types = ["TypeA", "TypeB", "TypeC", "TypeD", "TypeE", "modified_model"]

for model_type in model_types:
    # Get filtered image filenames from CSV file
    filtered_filenames = get_filtered_image_paths(csv_dir, model_type, min_distance, max_distance)
    
    if len(filtered_filenames) == 0:
        print(f"Warning: No filtered images found for {model_type}, skipping...")
        continue
    
    model_dir = os.path.join(input_base_dir, model_type)
    if os.path.exists(model_dir):
        print(f"Processing {model_type}...")
        embeddings, paths = get_image_embeddings(model_dir, model, preprocess, device, filtered_filenames, batch_size=batch_size)
        if len(embeddings) > 0:
            output_file = os.path.join(output_dir, f"RN50-clip_features_{model_type}_filtered_{int(min_distance)}_{int(max_distance)}.npz")
            np.savez(output_file, embeddings=embeddings, paths=paths)
            print(f"Features saved to {output_file} (shape: {embeddings.shape}, {len(paths)} images)\n")
        else:
            print(f"Warning: No valid images found in {model_type}\n")
    else:
        print(f"Warning: Directory not found: {model_dir}\n")

print("All features extracted and saved.")

