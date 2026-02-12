"""
Dot-based visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
import gc
from scipy.ndimage import binary_dilation

# Definition of the boundary term B_{i,j}(t)
def boundary(u):
    """
    Compute the boundary mask B_{i,j}(t) defined as:
        B_{i,j}(t) = (1 - u_{i,j}(t))　min(1, u_{i-1,j}(t) + u_{i+1,j}(t) + u_{i,j-1}(t) + u_{i,j+1}(t))
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
def uv_simulation(alpha, beta, gamma, delta, dv, n, seed):
    """
    Simulate the time evolution of u_{i,j}(t) and v_{i,j}(t)
    according to the discrete forms of the governing equations:
    u_{i,j}(t+dt) = u_{i,j}(t) + B_{i,j}(t) * H(η_{i,j}(t) < dt * (α - β v_{i,j}(t))), (Eq. u)
    where H(x) is the Heaviside step function.
    v_{i,j}(t+dt) = dt * (γ S_{i,j}(t) - δ v_{i,j}(t)) + D_v Δv_{i,j}(t+dt),           (Eq. v)
    where (Type E): S_{i,j}(t) = B_{i,j}(t) + B^n_{i,j}(t).                            (Eq. S_E)

    Parameters
    ----------
    alpha, beta, gamma, delta : float
        Reaction parameters.
    dv : float
        Diffusion coefficient of v.

    Returns
    -------
    u : 2D ndarray
        Final state of u field.
    i : int
        The number of time steps completed.
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
    
    # Initialize fields
    u = uI.copy()
    v = np.zeros((grid_number, grid_number))

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

        # Update v according to Eq. (v)
        v = np.real(ifft2(fft2(dt * (gamma * S - delta * v) + v) * kvhatinverse))

        # Early stopping when the infection expands to ±10% from the domain center
        if (np.any(u[4*grid_number//10, :] == 1) or
            np.any(u[6*grid_number//10, :] == 1) or
            np.any(u[:,4*grid_number//10] == 1) or
            np.any(u[:, 6*grid_number//10] == 1)):
            print(f"Calculation stopped at iteration {t}")
            break

        if t == loopnumber - 1:
            print(loopnumber)

    return u, v, t

# ==============================================================
# Helper function: dot plot rendering
# ==============================================================
def save_dot_plot(u, filename):
    """
    Render a dot-based binary plot of the field u and save it as a PNG image.

    Parameters
    ----------
    u : 2D ndarray
        Binary field where u = 1 indicates infected sites.
    filename : str
        Path for saving the generated PNG file.
    """

    # Prepare grid coordinates
    x, y = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))
    x_flat = x.flatten()
    y_flat = y.flatten()
    u_flat = u.flatten()

    # Dot size: white dots where u = 1, invisible otherwise
    sizes = np.where(u_flat == 1, 1, 0)

    plt.figure(figsize=(6, 6))
    plt.scatter(
        x_flat, y_flat,
        s=sizes,
        c='white',
        edgecolors='none',
        alpha=1.0,
        marker='o'
    )

    # Black background and clean axes
    ax = plt.gca()
    ax.set_facecolor('black')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    # Save image
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

# ==============================================================
# Parameter settings
# ==============================================================
domain_size = 12.0                      # Domain size, size of the human cornea (in mm)
dx = 0.012                              # Spatial step, size of a corneal epithelial cell (in mm)
dt = 1.0                                # Time step, \approx 0.5$ hours; estimated from visual similarity to ex vivo lesion data
dv = 0.05                               # Diffusion coefficient of v
loopnumber = 10000                      # Maximum number of iterations
grid_number = round(domain_size / dx)   # Number of lattice points per side, the number of cells along one edge

alpha = 0.5 #Net infection drive coefficient
beta = 1.0 #Infection inhibition coefficient (cytokine-dependent; e.g., cellular sensitivity to cytokines)
gamma_values = np.arange(0.5, 10.01, 0.5) #Cytokine secretion rate
delta = 0.5 #Cytokine degradation rate
n = 10 #The number of layers producing cytokine
seed_values = list(range(10, 20))  #Random seeds for reproducibility


# Create output directory if it does not exist
import os
# Get the absolute path of the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create the target directory inside the script directory
output_dir = os.path.join(script_dir, "TypeE")
os.makedirs(output_dir, exist_ok=True)

# ==============================================================
# Main parameter loop
# ==============================================================
for gamma in gamma_values:
    for seed in seed_values:

        # Run simulation
        u, v, t = uv_simulation(alpha, beta, gamma, delta, dv, n, seed)
        # Save figure with descriptive filename
        filename = os.path.join(
            output_dir,
            f"plot_a{alpha:.3f}_b{beta:.3f}_g{gamma:.3f}_"
            f"d{delta:.3f}_Dv{dv:.3f}_iter{t}_seed{seed}.png"
        )
        
        # Save dot representation of u
        save_dot_plot(u, filename)

        # Explicitly delete temporary variables to release memory
        del u, v, t

        # Invoke the garbage collector to ensure complete memory cleanup
        gc.collect()

