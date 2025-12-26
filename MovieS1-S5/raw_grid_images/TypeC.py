"""
Generate a side-by-side movie for multiple gamma values
-------------------------------------------------------

This script runs the simulation for a sequence of gamma values and
constructs a horizontal video panel by placing each gamma-frame side by side.
Since each simulation stops at a different iteration, shorter runs are padded
by repeating their final frame so that all gamma panels have identical length.

Each panel also overlays the text "iteration = t" in the lower-right corner.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from scipy.fft import fft2, ifft2

# ----------------------------------------------------------------------
#  Boundary term
# ----------------------------------------------------------------------
def boundary(u):
    u_up    = np.roll(u, shift=-1, axis=0)
    u_down  = np.roll(u, shift= 1, axis=0)
    u_left  = np.roll(u, shift=-1, axis=1)
    u_right = np.roll(u, shift= 1, axis=1)
    mask = u_up + u_down + u_left + u_right
    mask = np.clip(mask, 0, 1)
    return (1 - u) * mask


# ----------------------------------------------------------------------
#  Simulation per gamma
# ----------------------------------------------------------------------
def uv_simulation(alpha, beta, gamma, delta, dv):
    np.random.seed(0)

    uI = np.zeros((grid_number, grid_number))
    uI[grid_number//2, grid_number//2] = 1  # center seed

    kv = np.zeros((grid_number, grid_number))
    kv[0, 0] = 1 + 4 * dt * dv / dx**2
    kv[1, 0] = kv[-1, 0] = kv[0, 1] = kv[0, -1] = -dt * dv / dx**2
    kvhat = fft2(kv)
    kvhatinverse = 1 / kvhat

    u = uI.copy()
    v = np.zeros_like(u)

    frames = []  # store full time-series

    for t in range(loopnumber):

        # store a copy of u for this timestep
        frames.append(u.copy())

        b = boundary(u)
        eta = np.random.rand(grid_number, grid_number)
        h = (b * (eta < dt * (alpha - beta * v))).astype(float)
        u = u + h

        S = b
        v = np.real(ifft2(fft2(dt * (gamma * S - delta * v) + v) * kvhatinverse))

        # stopping conditions
        if (np.any(u[4*grid_number//10, :] == 1) or
            np.any(u[6*grid_number//10, :] == 1) or
            np.any(u[:,4*grid_number//10] == 1) or
            np.any(u[:,6*grid_number//10] == 1)):
            print(f"gamma={gamma}: stopped at iteration {t}")
            break

    return frames, t


# ----------------------------------------------------------------------
# Parameter settings
# ----------------------------------------------------------------------
domain_size = 12.0
dx = 0.012
dt = 1.0
dv = 0.05
loopnumber = 10000
grid_number = round(domain_size / dx)

alpha = 0.5
beta = 1.0
gamma_values = np.arange(0.5, 6.501, 1.0)
delta = 0.5

# directories
script_dir = os.path.dirname(os.path.abspath(__file__))
movie_dir = script_dir
os.makedirs(movie_dir, exist_ok=True)


# ----------------------------------------------------------------------
# Run all gamma simulations and store frame sequences
# ----------------------------------------------------------------------
all_frames = []      # list of lists; each item is frames for a gamma
all_iters  = []      # stopping iterations

for gamma in gamma_values:
    frames, t_stop = uv_simulation(alpha, beta, gamma, delta, dv)
    all_frames.append(frames)
    all_iters.append(t_stop)

max_length = max(len(fr) for fr in all_frames)


# ----------------------------------------------------------------------
# Pad simulations to the same number of frames
# ----------------------------------------------------------------------
for i in range(len(all_frames)):
    last_frame = all_frames[i][-1]
    while len(all_frames[i]) < max_length:
        all_frames[i].append(last_frame.copy())


# ----------------------------------------------------------------------
# Create horizontal concatenated movie
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 4))

writer = FFMpegWriter(fps=100, bitrate=2000)
movie_path = os.path.join(movie_dir, "MovieS3_TypeC.mp4")

with writer.saving(fig, movie_path, dpi=150):

    for k in range(max_length):
        ax.clear()

        # Construct horizontal panel by concatenating images
        panel = np.hstack([all_frames[i][k] for i in range(len(gamma_values))])
        ax.imshow(panel, cmap="gray", vmin=0, vmax=1)

        # Add gamma labels above each segment
        for i, gamma in enumerate(gamma_values):
            xpos = (i * grid_number) + grid_number / 2
            ypos = -10   # slightly above the image panel
            ax.text(xpos, ypos, f"γ = {gamma}",
                    color='black', fontsize=10, ha='center', va='bottom')
            
        # Add iteration labels under each segment
        x_offset = grid_number
        for i, gamma in enumerate(gamma_values):
            iter_text = f"iteration = {min(k, all_iters[i])}"
            xpos = (i * grid_number) + grid_number - 5
            ypos = grid_number - 5
            ax.text(xpos, ypos, iter_text, color='yellow',
                    fontsize=8, ha='right', va='bottom')

        ax.set_axis_off()
        writer.grab_frame()

print(f"Movie saved → {movie_path}")
