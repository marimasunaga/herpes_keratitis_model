import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ------------------------------------------------------------
# Detect first local minimum by sign change of first derivative
# (derivative: negative → positive)
# ------------------------------------------------------------
def detect_first_minimum(smoothed):
    """
    Detect the first local minimum of smoothed curve,
    but only searching from node index >= 20.
    A local minimum is where derivative changes: negative → positive.
    """
    d1 = np.gradient(smoothed)

    start_idx = 20  # search only after node_20

    for i in range(start_idx + 1, len(d1)):
        if d1[i-1] < 0 and d1[i] > 0:
            return i

    # Fallback: global minimum (after 20)
    return int(np.argmin(smoothed[start_idx:]) + start_idx)



# ================================================================
# Directory settings
# ================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

csv_root = os.path.join(script_dir, "4process_csv")
output_root = os.path.join(script_dir, "5graph_distance_map")
os.makedirs(output_root, exist_ok=True)

# Target CSV files (TypeA–E + modified_model)
csv_files = [
    "TypeA_branches_processed.csv",
    "TypeB_branches_processed.csv",
    "TypeC_branches_processed.csv",
    "TypeD_branches_processed.csv",
    "TypeE_branches_processed.csv",
    "modified_model_branches_processed.csv",
]

# Map filename → plot folder name
label_map = {
    "TypeA_branches_processed.csv": "TypeA",
    "TypeB_branches_processed.csv": "TypeB",
    "TypeC_branches_processed.csv": "TypeC",
    "TypeD_branches_processed.csv": "TypeD",
    "TypeE_branches_processed.csv": "TypeE",
    "modified_model_branches_processed.csv": "modified_model",
}

# ================================================================
# Main loop
# ================================================================
for csv_name in csv_files:

    csv_path = os.path.join(csv_root, csv_name)
    model_label = label_map[csv_name]

    # Output dir for this model
    out_dir = os.path.join(output_root, model_label)
    os.makedirs(out_dir, exist_ok=True)

    # Load CSV
    df = pd.read_csv(csv_path)

    # Node columns (node_0, node_1, ...)
    node_cols = [col for col in df.columns if col.startswith("node_")]
    df[node_cols] = df[node_cols].apply(pd.to_numeric, errors="coerce")

    print(f"Processing {csv_name} ({model_label})...")

    # ===========================================================
    # For each branch
    # ===========================================================
    for branch_id, row in df.iterrows():

        distance_values = row[node_cols].dropna().astype(float).values

        if len(distance_values) == 0:
            continue

        node_0 = distance_values[0]

        # ---- Condition: node_0 must be in 10–20 ----
        if not (10 <= node_0 < 20):
            continue

        x = np.arange(len(distance_values))

        # Gaussian smoothing
        sigma = 5
        smoothed = gaussian_filter1d(distance_values, sigma=sigma)

        # Detect first local minimum (root)
        root_idx = detect_first_minimum(smoothed)

        # Compute root value as mean of ±5 around the detected minimum
        left = max(0, root_idx - 5)
        right = min(len(smoothed), root_idx + 5)
        root_value = np.mean(smoothed[left:right])

        # =======================================================
        # Plot
        # =======================================================
        plt.figure(figsize=(6, 4))

        # Original data
        plt.plot(
            x, distance_values,
            "o-", color="black", markersize=3, alpha=0.4,
            label="Original data"
        )

        # Smoothed curve
        plt.plot(
            x, smoothed,
            "-", color="blue", linewidth=2,
            label=f"Gaussian smoothing (σ={sigma})"
        )

        # --- Mark first local minimum ---
        plt.scatter(root_idx, smoothed[root_idx],
                    color="blue", s=30, label="First local minimum (node ≥ 20)")

        # ---- root region (±5 nodes around first local minimum) ----
        root_left  = max(0, root_idx - 5)
        root_right = min(len(distance_values), root_idx + 6)  # +6 because right is exclusive

        plt.axvspan(
            root_left,
            root_right,
            color="lightblue",
            alpha=0.35,
            label="Root region (minumun ±5 nodes)"
        )
        
        # --- Mark Tip resion ---
        L = len(distance_values)
        tip_start = int(L * 0.8)
        tip_end   = int(L * 0.9)
        plt.axvspan(
            tip_start,
            tip_end,
            color="orange",
            alpha=0.3,
            label="Tip region (last 10–20%)"
        )

        # Title with root value
        plt.title(f"Branch {branch_id}")

        plt.xlabel("Node index along the branch")
        plt.ylabel("Distance map value")

        plt.grid(True)
        plt.legend()

        # Save
        out_path = os.path.join(out_dir, f"branch_{branch_id}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"  Saved → {out_path}")

print("All graphs generated successfully.")
