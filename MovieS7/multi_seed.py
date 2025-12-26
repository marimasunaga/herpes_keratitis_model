"""
Multi-seed simulation of reaction-diffusion system
This script simulates the dynamics of u and v fields with multiple initial seeds.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
import imageio.v2 as imageio
import os
import matplotlib.cm as cm
import matplotlib.colors as colors


# ==============================================================
# Boundary term calculation
# ==============================================================
def boundary(u):
    """
    Calculate boundary mask for growth probability.
    
    Parameters:
    -----------
    u : ndarray
        Binary field (0 or 1) representing the current state
        
    Returns:
    --------
    boundary_mask : ndarray
        Mask indicating boundary regions where growth can occur
    """
    u_up    = np.roll(u, shift=-1, axis=0)
    u_down  = np.roll(u, shift= 1, axis=0)
    u_left  = np.roll(u, shift=-1, axis=1)
    u_right = np.roll(u, shift= 1, axis=1)

    boundary_mask = u_up + u_down + u_left + u_right
    boundary_mask = np.clip(boundary_mask, 0, 1)
    boundary_mask = (1 - u) * boundary_mask
    return boundary_mask


# ==============================================================
# Main simulation function
# ==============================================================
def uv_simulation(alpha, beta, gamma, delta, dv):
    """
    Simulate the reaction-diffusion system for u and v fields.
        
    Returns:
    --------
    u_frames : list
        List of u field snapshots at each time step
    v_frames : list
        List of v field snapshots at each time step
    """
    np.random.seed(0)

    # Initialize u field with multiple seeds
    uI = np.zeros((grid_number, grid_number))
    uI[grid_number//2, grid_number//2] = 1
    uI[3*grid_number//8, 3*grid_number//8] = 1
    uI[3*grid_number//8, 5*grid_number//8] = 1
    uI[5*grid_number//8, 3*grid_number//8] = 1
    uI[5*grid_number//8, 5*grid_number//8] = 1

    # Precompute FFT kernel for v field diffusion
    kv = np.zeros((grid_number, grid_number))
    kv[0, 0] = 1 + 4 * dt * dv / dx**2
    kv[1, 0] = kv[-1, 0] = kv[0,1] = kv[0,-1] = -dt * dv / dx**2

    kvhat = fft2(kv)
    kvhatinverse = 1 / kvhat

    # Initialize fields
    u = uI.copy()
    v = np.zeros((grid_number, grid_number))

    u_frames = []
    v_frames = []

    # Time evolution
    for i in range(loopnumber):
        # Calculate boundary and growth probability
        b = boundary(u)
        eta = np.random.rand(grid_number, grid_number)

        # Update u field (stochastic growth)
        h = (b * (eta < dt * (alpha - beta * v))).astype(float)
        u = u + h

        # Update v field (reaction-diffusion)
        S = u
        v = np.real(ifft2(fft2(dt * (gamma * S - delta * v) + v) * kvhatinverse))

        # Store frames
        u_frames.append(u.copy())
        v_frames.append(v.copy())

    return u_frames, v_frames


# ==============================================================
# Parameter settings
# ==============================================================
domain_size = 12.0      # Domain size
dx = 0.012              # Spatial discretization
dt = 1.0                # Time step
dv = 0.05               # Diffusion coefficient for v field
loopnumber = 1000       # Number of time steps
grid_number = round(domain_size / dx)

# Reaction-diffusion parameters
alpha = 0.5            
beta = 1.0            
gamma = 3.0             
delta = 0.5            

# ==============================================================
# Output directory setup
# ==============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = script_dir
os.makedirs(output_dir, exist_ok=True)

# ==============================================================
# Run simulation
# ==============================================================
u_frames, v_frames = uv_simulation(alpha, beta, gamma, delta, dv)

# ==============================================================
# Visualization: Save side-by-side frames
# ==============================================================
frame_files = []

# Contour plot settings for v field
vmin, vmax = 0.0, 1.0
levels = np.linspace(vmin, vmax, 50)
norm = colors.Normalize(vmin=vmin, vmax=vmax)

for idx, (u, v) in enumerate(zip(u_frames, v_frames)):
    fname = os.path.join(output_dir, f"frame_{idx:04d}.png")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Left panel: u field (grayscale)
    axs[0].imshow(u, cmap='gray')
    axs[0].set_title("u")
    axs[0].axis('off')

    # Right panel: v field (contour representation)
    cs = axs[1].contour(v, levels=levels, cmap='viridis', linewidths=2.0)
    axs[1].set_title(f"v (iter={idx})")
    axs[1].axis('off')

    # Add colorbar
    sm = cm.ScalarMappable(norm=norm, cmap='viridis')
    sm.set_array([])
    fig.colorbar(sm, ax=axs[1], fraction=0.046, pad=0.04)

    # Save frame
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close(fig)

    frame_files.append(fname)

# ==============================================================
# Create movie from frames
# ==============================================================
movie_path = os.path.join(output_dir, "movie_S7.mp4")

with imageio.get_writer(movie_path, fps=100) as writer:
    for fname in frame_files:
        img = imageio.imread(fname)
        writer.append_data(img)

print("Movie saved:", movie_path)

# ==============================================================
# Clean up: Delete frame images after movie creation
# ==============================================================
for fname in frame_files:
    if os.path.exists(fname):
        os.remove(fname)

print("Frame images deleted.")
