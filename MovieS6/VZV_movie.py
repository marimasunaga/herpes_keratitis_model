"""
VZV movie generation
This script simulates the dynamics of u and v fields and creates a side-by-side movie.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
import imageio.v2 as imageio
import os
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy.ndimage import binary_dilation

# Definition of the boundary term B_{i,j}(t)
def boundary(u):
    """
    Compute the boundary mask B_{i,j}(t) defined as:
        B_{i,j}(t) = min(1, u_{i-1,j}(t) + u_{i+1,j}(t) + u_{i,j-1}(t) + u_{i,j+1}(t))
    where u is a 2D binary field (0 or 1).
    """
    # Shift u in four directions (up, down, left, right)
    u_up    = np.roll(u, shift=-1, axis=0)
    u_down  = np.roll(u, shift= 1, axis=0)
    u_left  = np.roll(u, shift=-1, axis=1)
    u_right = np.roll(u, shift= 1, axis=1)

    # Compute the sum of the four neighboring cells
    boundary_mask = u_up + u_down + u_left + u_right

    # Clip the values to obtain B_{i,j}(t) = min(1, sum)
    boundary_mask = np.clip(boundary_mask, 0, 1)

    # Apply boundary mask only to uninfected cells 
    boundary_mask = (1 - u) * boundary_mask

    return boundary_mask

def boundary_n(u, n):
    """
    Compute B^{(n)}_{i,j}(t) according to the definition:
        B^{(n)}_{i,j}(t) = u_{i,j}(t) * min(1, sum_{1 <= |k|+|l| <= n} (1 - u_{i+k,j+l}(t)) )

    Parameters
    ----------
    u : 2D numpy array
        Binary array representing infected (1) and uninfected (0) cells.
    n : int
        Neighborhood radius (Manhattan distance).

    Returns
    -------
    Bn : 2D numpy array
        Boundary mask indicating infected cells adjacent (within n) to uninfected cells.
    """

    # Compute mask of uninfected cells (1 - u)
    uninfected = 1 - u

    # Dilate uninfected region by n (Manhattan neighborhood)
    # This marks all cells within distance n of an uninfected cell
    # Structure defines 4-neighborhood (|k| + |l| = 1)
    structure = np.array([[0,1,0],
                          [1,1,1],
                          [0,1,0]])
    dilated_uninfected = binary_dilation(uninfected, structure=structure, iterations=n).astype(int)

    # B^{(n)}_{i,j}(t) = u_{i,j}(t) * min(1, dilated_uninfected)
    # → infected cells that are within n steps of uninfected cells
    Bn = u * np.clip(dilated_uninfected, 0, 1)

    return Bn

# Simulation of u and v fields
def uv_simulation(alpha, beta, gamma, delta, dv, dw, C, n, seed):
    """
    Simulate the time evolution of u_{i,j}(t) and v_{i,j}(t)
    according to the discrete forms of the governing equations:
    u_{i,j}(t+dt) = u_{i,j}(t) + B_{i,j}(t) * H(η_{i,j}(t) < dt * (α - β v_{i,j}(t))), (Eq. u)
    where H(x) is the Heaviside step function.
    v_{i,j}(t+dt) = dt * (γ S_{i,j}(t) - δ v_{i,j}(t) + w_{i,j}(t+dt)) + D_v Δv_{i,j}(t+dt),           (Eq. v)
    where (Type E): S_{i,j}(t) = B_{i,j}(t) + B^n_{i,j}(t).                            (Eq. S_E)
    w_{i,j}(t+dt) = dt * C + D_w Δw_{i,j}(t+dt) (if (i = 0 or -1) or (j = 0 or -1)), 
                    D_w Δw_{i,j}(t+dt) (otherwise).                                    (Eq. w)

    Parameters
    ----------
    alpha, beta, gamma, delta : float
        Reaction parameters.
    dv : float
        Diffusion coefficient of v.
    dw : float
        Diffusion coefficient of w.
    C : float
        Immune-cell-derived cytokine secretion rate.
    n : int
        The number of layers producing cytokine.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    u_frames : list
        List of u field snapshots at each time step
    v_frames : list
        List of v field snapshots at each time step
    """
    # Fix random seed for reproducibility
    np.random.seed(seed)

    # Initialize u field with zeros
    uI = np.zeros((grid_number, grid_number))

    # Initial condition: one infected cell at the center
    uI[grid_number//2, grid_number//2] = 1

    # Define the discrete diffusion kernel for v (Laplace operator)
    kv=np.zeros((grid_number, grid_number))
    kv[0, 0] = 1 + 4 * dt * dv / dx**2
    kv[1, 0] = kv[-1, 0] = kv[0, 1] = kv[0, -1] = -dt * dv / dx**2

    # Compute the Fourier transform of the kernel
    kvhat = fft2(kv)
    kvhatinverse = 1 / kvhat

    # Define the discrete diffusion kernel for w (Laplace operator)
    kw=np.zeros((grid_number, grid_number))
    kw[0, 0] = 1 + 4 * dt * dw / dx**2
    kw[1, 0] = kw[-1, 0] = kw[0, 1] = kw[0, -1] = -dt * dw / dx**2

    # Compute the Fourier transform of the kernel
    kwhat = fft2(kw)
    kwhatinverse = 1 / kwhat
    
    # Initialize fields
    u = uI.copy()
    v = np.zeros((grid_number, grid_number))
    w = np.zeros((grid_number, grid_number))

    u_frames = []
    v_frames = []

    # --- Time iteration ---
    for t in range(loopnumber):
        # Boundary term B_{i,j}(t) as defined in Eq. (B)
        b = boundary(u)
        bn = boundary_n(u, n)

        # Update rule for u_{i,j}(t):
        eta = np.random.rand(grid_number, grid_number)
        h = (b * (eta < dt * (alpha - beta * v))).astype(float)
        u = u + h
        # Cytokine source term (Type E): S_{i,j}(t) = B_{i,j}(t) + B^n_{i,j}(t)
        S = b + bn

        # Update v and w according to Eq. (v) and Eq. (w)
        v = np.real(ifft2(fft2(dt * (gamma * S - delta * v) + v) * kvhatinverse))
        
        w[0,:] += C
        w[-1,:] += C
        w[:,0] += C
        w[:,-1] += C
        w = np.real(ifft2(fft2(dt * w) * kwhatinverse))

        v += w

        # Store frames
        u_frames.append(u.copy())
        v_frames.append(v.copy())

        # Early stopping if the infection reaches the domain boundary
        if (np.any(u[grid_number//10, :] == 1) or
            np.any(u[9*grid_number//10, :] == 1) or
            np.any(u[:, grid_number//10] == 1) or
            np.any(u[:, 9*grid_number//10] == 1)):
            print(f"Calculation stopped at iteration {t}")
            break

        if t == loopnumber - 1:
            print(loopnumber)

    return u_frames, v_frames


# ==============================================================
# Parameter settings
# ==============================================================
domain_size = 12.0                      # Domain size, size of the human cornea (in mm)
dx = 0.012                              # Spatial step, size of a corneal epithelial cell (in mm)
dt = 1.0                                # Time step, \approx 0.5$ hours; estimated from visual similarity to ex vivo lesion data
dv = 0.05                               # Diffusion coefficient of cytokine
dw = 50                                 # Effective diffusion coefficient representing immune-cell derived cytokine
loopnumber = 1500                      # Maximum number of iterations
grid_number = round(domain_size / dx)   # Number of lattice points per side, the number of cells along one edge

alpha = 0.5 #Net infection drive coefficient
beta = 1.0 #Infection inhibition coefficient (cytokine-dependent; e.g., cellular sensitivity to cytokines)
gamma = 1.5 #Cytokine secretion rate
delta = 0.5 #Cytokine degradation rate
n = 10 #The number of layers producing cytokine
C = 0.03 #Immune-cell-derived cytokine secretion rate
seed = 0  # Random seed for reproducibility

# ==============================================================
# Output directory setup
# ==============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = script_dir
os.makedirs(output_dir, exist_ok=True)

# ==============================================================
# Run simulation
# ==============================================================
u_frames, v_frames = uv_simulation(alpha, beta, gamma, delta, dv, dw, C, n, seed)

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
    cs = axs[1].contour(v, levels=levels, cmap='viridis', linewidths=0.5)  # Thinner lines for finer contours
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
movie_path = os.path.join(output_dir, "movie_S6.mp4")

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

