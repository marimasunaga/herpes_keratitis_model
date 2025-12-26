import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, ttest_rel, wilcoxon
from statsmodels.stats.power import TTestPower
from numpy import mean, std
import os
from scipy.ndimage import gaussian_filter1d

# ================================================================
# Utility: detect first minimum after start_idx
# ================================================================
def detect_first_minimum(smoothed, start_idx=20):
    """
    Detect the first local minimum after start_idx.
    A minimum is detected when derivative changes: negative → positive.
    """
    d1 = np.gradient(smoothed)

    for i in range(start_idx + 1, len(d1)):
        if d1[i-1] < 0 and d1[i] > 0:
            return i

    # fallback: choose global minimum after start_idx
    return int(np.argmin(smoothed[start_idx:]) + start_idx)

# ================================================================
# Directory settings
# ================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

csv_root = os.path.join(script_dir, "4process_csv")  # input directory
output_root = os.path.join(script_dir, "6stat_graph")  # output directory
os.makedirs(output_root, exist_ok=True)

boxplot_path = os.path.join(output_root, "boxplot.png")
summary_path = os.path.join(output_root, "summary.txt")

# Collect all CSV files under 4process_csv
csv_paths = sorted([os.path.join(csv_root, f) for f in os.listdir(csv_root) if f.endswith(".csv")])

# Mapping from CSV filename to desired label
label_map = {
    "TypeA_branches_processed": "TypeA",
    "TypeB_branches_processed": "TypeB",
    "TypeC_branches_processed": "TypeC",
    "TypeD_branches_processed": "TypeD",
    "TypeE_branches_processed": "TypeE",
    "modified_model_branches_processed": "ModifiedModel"
}

# ================================================================
# Analysis settings
# ================================================================
sigma = 5  # smoothing

all_plot_data = []
summary_table = []

# ================================================================
# Main analysis loop
# ================================================================
for path in csv_paths:
    df = pd.read_csv(path, index_col=0)

    data_tip = []
    data_root = []

    raw_name = os.path.splitext(os.path.basename(path))[0]
    model_name = label_map.get(raw_name, raw_name)

    print(f"Processing {model_name} ...")

    for _, row in df.iterrows():
        values = pd.to_numeric(row, errors='coerce').dropna().values
        if len(values) == 0:
            continue

        node_0 = values[0]
        if not (10 <= node_0 < 20):
            continue

        # --- smooth values ---
        smoothed = gaussian_filter1d(values, sigma=sigma)
        # --- detect first minimum ---
        min_idx = detect_first_minimum(smoothed, start_idx=20)

        # root = average over ±5 nodes around the minimum
        start = max(0, min_idx - 5)
        end   = min(len(smoothed), min_idx + 5)
        root = np.mean(values[start:end])

        # Length of the distance-map array
        L = len(values)

        # Tip region defined as a percentage of the tail portion (last 10%–20%)
        tip_start_ratio = 0.10   # 10%
        tip_end_ratio   = 0.20   # 20%

        # Convert ratios into actual index positions
        tip_start_idx = int(L * tip_start_ratio)   # 10% from the end
        tip_end_idx   = int(L * tip_end_ratio)     # 20% from the end

        # Slice the array using negative indexing (Python convention)
        tip_region = values[-tip_end_idx : -tip_start_idx]

        # Skip if the selected region is empty (safety check)
        if len(tip_region) == 0:
            continue

        # Compute mean value of the tip region
        tip = np.mean(tip_region)

        data_root.append(root)
        data_tip.append(tip)

    data_root = np.array(data_root)
    data_tip = np.array(data_tip)

    if len(data_root) < 3:
        continue

    if model_name == "ModifiedModel":
        # expect  root < tip
        diff = data_root - data_tip  # negative on average is expected
    else:
        # expect  tip < root
        diff = data_tip - data_root

    stat, p = shapiro(diff)
    is_normal = p >= 0.05
    normality_result = "Yes" if is_normal else "No"

    # Paired test
    if model_name == "ModifiedModel":
        # Null hypothesis: root < tip
        if is_normal:
            t_stat, p_sig = ttest_rel(data_root, data_tip, alternative='greater')
            method_used = "paired t-test"
        else:
            w_stat, p_sig = wilcoxon(data_root, data_tip, alternative='greater', zero_method='wilcox')
            method_used = "Wilcoxon signed-rank"
    else:
        # Null hypothesis: tip < root
        if is_normal:
            t_stat, p_sig = ttest_rel(data_tip, data_root, alternative='greater')
            method_used = "paired t-test"
        else:
            w_stat, p_sig = wilcoxon(data_tip, data_root, alternative='greater', zero_method='wilcox')
            method_used = "Wilcoxon signed-rank"

    # Significance
    if p_sig < 0.001:
        sig = "***"
    elif p_sig < 0.01:
        sig = "**"
    elif p_sig < 0.05:
        sig = "*"
    else:
        sig = "ns"

    effect_size = abs(mean(diff) / std(diff, ddof=1))

    analysis = TTestPower()

    if model_name == "ModifiedModel":
        alt = "larger"  # Match the alternative='greater' in the test
    else:
        alt = "larger"  

    try:
        power = analysis.power(effect_size=effect_size, nobs=len(diff), alpha=0.05, alternative=alt)
    except:
        power = np.nan

    # Plot data
    temp_df = pd.DataFrame({
        "Value": np.concatenate([data_root, data_tip]),
        "Region": ["root"] * len(data_root) + ["tip"] * len(data_tip),
        "Model": [model_name] * (len(data_tip) + len(data_root)),
        "Sample": list(range(len(data_root))) + list(range(len(data_tip)))
    })

    all_plot_data.append(temp_df)

    summary_table.append({
        "Model": model_name,
        "N": len(diff),
        "Effect Size (d)": effect_size,
        "Power": power,
        "P-value": p_sig,
        "Significance": sig,
        "Normality": normality_result,
        "Method": method_used
    })

# ================================================================
# Visualization: boxplot + stripplot
# ================================================================
summary_df = pd.DataFrame(summary_table)
plot_df = pd.concat(all_plot_data, ignore_index=True)

plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
ax = plt.gca()

sns.boxplot(data=plot_df, x="Model", y="Value", hue="Region",
            palette="pastel", fliersize=0, ax=ax)

handles, labels = ax.get_legend_handles_labels()
box_legend = ax.legend(handles, labels, loc="upper right", fontsize=17)

sns.stripplot(data=plot_df, x="Model", y="Value", hue="Region",
              dodge=True, palette=["black"], size=3, alpha=0.3, ax=ax)

ax.legend_.remove()
ax.add_artist(box_legend)

# Add significance markers
y_max = plot_df["Value"].max()
y_offset = 0.5

for i, row in summary_df.iterrows():
    model = row["Model"]
    sig = row["Significance"]

    x1, x2 = i - 0.2, i + 0.2

    y = plot_df[plot_df["Model"] == model]["Value"].max() + y_offset

    plt.plot([x1, x1, x2, x2], [y, y + 0.2, y + 0.2, y], lw=1.5, color='black')
    plt.text(i, y + 0.25, sig, ha='center', va='bottom', fontsize=17)

plt.ylim(0, y_max + 2.5)
plt.ylabel("Distance Map (Average)", fontsize=17)
plt.xlabel("")
# Set axis tick label font sizes explicitly
ax.tick_params(axis='x', labelsize=17)
ax.tick_params(axis='y', labelsize=17)
plt.tight_layout()

plt.savefig(boxplot_path, dpi=300)
plt.close()

print(f"Boxplot saved → {boxplot_path}")

# ================================================================
# Save summary
# ================================================================
with open(summary_path, "w") as f:
    f.write("Statistical Summary\n====================\n\n")
    f.write(summary_df.to_string(index=False))

print(f"Summary saved → {summary_path}")

# ============================================================
# 2. Boxplot with paired lines
# ============================================================
subdirs = summary_df["Model"].tolist()

plt.figure(figsize=(12, 6))
sns.boxplot(data=plot_df, x="Model", y="Value", hue="Region",
            palette="pastel", fliersize=0)

ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], loc="upper right", fontsize=17)

# ---- Draw paired lines ----
for model in subdirs:
    df_m = plot_df[plot_df["Model"] == model]

    for sample in df_m["Sample"].unique():
        df_s = df_m[df_m["Sample"] == sample]
        if len(df_s) != 2:
            continue

        # --- swapped x positions: root(left), tip(right) ---
        x_left = subdirs.index(model) - 0.2   # root
        x_right = subdirs.index(model) + 0.2  # tip

        y_root = df_s[df_s["Region"] == "root"]["Value"].values[0]
        y_tip = df_s[df_s["Region"] == "tip"]["Value"].values[0]

        # Color rule still valid: red = upward, blue = downward
        color = "red" if y_tip > y_root else "blue"

        plt.plot([x_left, x_right], [y_root, y_tip],
                 color=color, alpha=0.7, linewidth=1.2)

plt.title("Paired Tip–Root Change per Branch", fontsize=20)
plt.ylabel("Distance Map (Average)", fontsize=17)
plt.xlabel("")
# Set axis tick label font sizes explicitly
ax.tick_params(axis='x', labelsize=17)
ax.tick_params(axis='y', labelsize=17)
plt.tight_layout()

# Save paired-line plot
pairplot_path = os.path.join(output_root, "boxplot_with_pairlines.png")
plt.savefig(pairplot_path, dpi=300)
plt.close()

print(f"Paired-line plot saved → {pairplot_path}")
