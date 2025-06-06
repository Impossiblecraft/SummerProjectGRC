# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 01:19:58 2025

@author: Rajveer Daga
"""

import numpy as np
import matplotlib.pyplot as plt

# Spiral arm parameters (Reid+ 2019)
spiral_arms = {
    "Sagittarius": {"p": 7.3, "R_ref": 5.7, "theta_ref": 27.0},
    "Perseus": {"p": 9.4, "R_ref": 9.9, "theta_ref": 50.0},
    "Scutum": {"p": 19.8, "R_ref": 5.0, "theta_ref": 20.0},
    "Outer": {"p": 13.8, "R_ref": 13.0, "theta_ref": 77.0},
    "Local Spur": {"p": 11.5, "R_ref": 8.2, "theta_ref": 0.0},  # Approximate
}

# Solar position
R_sun = 8.2

# Solar neighborhood plot bounds
x_min, x_max = -10, -6.5
y_min, y_max = -2, 2

# Generate spiral arm within visible range
def generate_spiral_clipped(p_deg, R_ref, theta_ref_deg, R_min=4, R_max=15, n_points=2000):
    p = np.radians(p_deg)
    theta_ref = np.radians(theta_ref_deg)
    R_vals = np.linspace(R_min, R_max, n_points)
    theta_vals = theta_ref + (1 / np.tan(p)) * np.log(R_vals / R_ref)
    X = R_vals * np.cos(theta_vals)
    Y = R_vals * np.sin(theta_vals)

    # Keep only points inside plot bounds
    mask = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)
    return X[mask], Y[mask]

# Plotting
fig, ax = plt.subplots(figsize=(10, 10))

# Overlay spiral arms (clipped)
for name, params in spiral_arms.items():
    X, Y = generate_spiral_clipped(
        p_deg=params["p"],
        R_ref=params["R_ref"],
        theta_ref_deg=params["theta_ref"]
    )
    ax.plot(X, Y, label=name)
# Mark the Sun's position
ax.plot(-R_sun, 0, 'yo', label='Sun')

# Set up axis limits and labels
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel("X (kpc)")
ax.set_ylabel("Y (kpc)")
ax.set_title("Spiral Arms in the Solar Neighborhood")
ax.set_aspect('equal')
ax.legend()
plt.grid(True)
plt.tight_layout()
plt.show()