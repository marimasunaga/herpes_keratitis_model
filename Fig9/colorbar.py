import os
import matplotlib.pyplot as plt
import matplotlib as mpl

# Colormap (example: viridis)
cmap = plt.cm.viridis
norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

# Create a thinner figure
fig, ax = plt.subplots(figsize=(0.4, 4))  # width reduced

# Draw colorbar only
cb = mpl.colorbar.ColorbarBase(
    ax,
    cmap=cmap,
    norm=norm,
    orientation='vertical'
)

# Explicitly set ticks from 0 to 1.0
cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
cb.set_ticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Save the figure in the same directory as the script
output_path = os.path.join(script_dir, "colorbar_0_1.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.close()