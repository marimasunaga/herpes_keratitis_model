
import numpy as np
import matplotlib.pyplot as plt
import gc
import imageio

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

# Simulation of u field
def u_simulation(alpha, frame_callback=None):
    """
    Simulate the time evolution of u_{i,j}(t)
    according to the discrete form of the governing equation:
    u_{i,j}(t+dt) = u_{i,j}(t) + B_{i,j}(t) * H(η_{i,j}(t) < dt * α * ζ_{i,j}),
    where H(x) is the Heaviside step function.

    Parameters
    ----------
    alpha : float
        Reaction parameter.
    frame_callback : callable, optional
        Function to call at each time step with (u, t) as arguments.

    Returns
    -------
    u : 2D ndarray
        Final state of u field.
    t : int
        The number of time steps completed.
    """
    # Fix random seed for reproducibility
    np.random.seed(0)

    # Initialize u field with zeros
    uI = np.zeros((grid_number, grid_number))

    # Initial condition: one infected cell at the center
    uI[grid_number//2, grid_number//2] = 1
    
    # Initialize fields
    u = uI.copy()

    # Initialize zeta: ζ_{i,j} = 0.5 + ξ_{i,j}, where ξ_{i,j} ~ U(0,1)
    xi = np.random.rand(grid_number, grid_number)
    zeta = 0.5 + xi

    # Call callback for initial state
    if frame_callback is not None:
        frame_callback(u, 0)

    # --- Time iteration ---
    for t in range(1, loopnumber + 1):
        # Boundary term B_{i,j}(t) as defined in Eq. (B)
        b = boundary(u)

        # Update rule for u_{i,j}(t):
        eta = np.random.rand(grid_number, grid_number)
        h = (b * (eta < dt * alpha * zeta)).astype(float)
        u = u + h

        # Call callback for current frame
        if frame_callback is not None:
            frame_callback(u, t)

        # Early stopping if the infection reaches the domain boundary
        if (np.any(u[4*grid_number//10, :] == 1) or
            np.any(u[6*grid_number//10, :] == 1) or
            np.any(u[:,4*grid_number//10] == 1) or
            np.any(u[:, 6*grid_number//10] == 1)):
            print(f"Calculation stopped at iteration {t}")
            break

        if t == loopnumber:
            print(loopnumber)

    return u, t

# ==============================================================
# Helper function: grid plot rendering
# ==============================================================
def create_frame(u, t, alpha):
    """
    Create a frame with the field u and display iteration and alpha values.

    Parameters
    ----------
    u : 2D ndarray
        Binary field where u = 1 indicates infected sites.
    t : int
        Current iteration number.
    alpha : float
        Reaction parameter.

    Returns
    -------
    frame : numpy array
        Image array of the frame.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(u, cmap='gray', origin='upper', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add text with iteration and alpha values
    text = f'iteration: {t}\nα = {alpha:.3f}'
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Convert figure to numpy array
    fig.canvas.draw()
    # Get the RGBA buffer and convert to RGB
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    # Convert RGBA to RGB
    frame = frame[:, :, :3]
    plt.close(fig)
    
    return frame

# ==============================================================
# Parameter settings
# ==============================================================
domain_size = 12.0                      # Domain size, size of the human cornea (in mm)
dx = 0.012                              # Spatial step, size of a corneal epithelial cell (in mm)
dt = 1.0                                # Time step, \approx 0.5$ hours; estimated from visual similarity to ex vivo lesion data
loopnumber = 10000                       # Maximum number of iterations
grid_number = round(domain_size / dx)   # Number of lattice points per side, the number of cells along one edge

alpha = 0.01 #Net infection drive coefficient

# Create output directory if it does not exist
import os
# Get the absolute path of the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create the target directory inside the script directory
output_dir = script_dir
os.makedirs(output_dir, exist_ok=True)

# ==============================================================
# Main simulation
# ==============================================================
frames = []

def save_frame(u, t):
    """Callback function to save each frame"""
    frame = create_frame(u, t, alpha)
    frames.append(frame)

# Run simulation with frame callback
u, t = u_simulation(alpha, frame_callback=save_frame)

# Save movie
movie_filename = os.path.join(
    output_dir,
    f"movie_a{alpha:.3f}.mp4"
)

# Create movie from frames
if frames:
    try:
        # Try with ffmpeg codec
        imageio.mimwrite(movie_filename, frames, fps=100, codec='libx264', quality=8)
    except Exception as e:
        # Fallback to default settings
        print(f"Warning: Could not use libx264, trying default codec: {e}")
        imageio.mimwrite(movie_filename, frames, fps=100)
    print(f"Movie saved: {movie_filename} ({len(frames)} frames)")

# Clean up
del u, t, frames

# Invoke the garbage collector to ensure complete memory cleanup
gc.collect()

