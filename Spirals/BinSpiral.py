import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

dat=pd.read_csv("Spiralprotv.csv")

#Spirals overlaid incorrectly, otherwise right
#Rotate daat to fit it maybe?? Maybe issue with that

# Extract position and rotational velocity data
x = dat['px'].values  # x position in kpc
y = dat['py'].values  # y position in kpc
rotv = np.abs(dat['rotv'].values)  # absolute rotational velocity

print(f"Data shape: {dat.shape}")
print(f"Position range: x = [{x.min():.2f}, {x.max():.2f}] kpc, y = [{y.min():.2f}, {y.max():.2f}] kpc")
print(f"Rotational velocity range: [{rotv.min():.2f}, {rotv.max():.2f}] km/s")

# Define Sun's position in the Galaxy (standard solar position)
sun_x = -8.2  # kpc
sun_y = 0.0  # kpc

# Filter to keep only stars within a 2kpc×2kpc square centered on the Sun
square_size = 2  # kpc (half-width of the square)
within_square_mask = (
    (x >= sun_x - square_size) & 
    (x <= sun_x + square_size) & 
    (y >= sun_y - square_size) & 
    (y <= sun_y + square_size)
)

# Apply filter
x = x[within_square_mask]
y = y[within_square_mask]
rotv = rotv[within_square_mask]

print(f"Filtered data shape: {within_square_mask.sum()} stars within 2kpc×2kpc square around Sun")
print(f"Square region: X=[{sun_x-square_size:.1f}, {sun_x+square_size:.1f}] kpc, Y=[{sun_y-square_size:.1f}, {sun_y+square_size:.1f}] kpc")

# Convert kpc to pc
x_pc = x * 1000  # 1 kpc = 1000 pc
y_pc = y * 1000

# Define bin size in pc
bin_size = 10  # pc

# Create bins
# Determine bin edges for x and y
x_min, x_max = np.floor(x_pc.min()), np.ceil(x_pc.max())
y_min, y_max = np.floor(y_pc.min()), np.ceil(y_pc.max())

# Create bin edges with 2pc spacing
x_bins = np.arange(x_min, x_max + bin_size, bin_size)
y_bins = np.arange(y_min, y_max + bin_size, bin_size)

print(f"Creating {len(x_bins)-1}×{len(y_bins)-1} bins of size {bin_size}pc × {bin_size}pc")

# Create 2D histogram with mean rotational velocity
H, xedges, yedges = np.histogram2d(x_pc, y_pc, bins=[x_bins, y_bins], weights=rotv)
count, _, _ = np.histogram2d(x_pc, y_pc, bins=[x_bins, y_bins])

# Avoid division by zero
mask = count > 0
mean_rotv = np.zeros_like(H)
mean_rotv[mask] = H[mask] / count[mask]

# Convert bin edges back to kpc for plotting
xedges_kpc = xedges / 1000
yedges_kpc = yedges / 1000

# Create a meshgrid for plotting
X, Y = np.meshgrid(xedges_kpc[:-1], yedges_kpc[:-1])

# Set up the plot
plt.figure(figsize=(12, 10))

# Create color map of binned rotational velocities
# Use pcolormesh for better performance with large grids
mesh = plt.pcolormesh(xedges_kpc, yedges_kpc, mean_rotv.T, 
                      shading='flat', cmap='viridis')
                      
# Add colorbar
cbar = plt.colorbar(mesh)
cbar.set_label('Mean Rotational Velocity (km/s)', size=12)

# Add grid
plt.grid(False)

# Add labels and title
plt.xlabel('X (kpc)', fontsize=14)
plt.ylabel('Y (kpc)', fontsize=14)
plt.title('Binned Rotational Velocities (5pc×5pc bins)', fontsize=16)

# Set equal aspect ratio
plt.axis('equal')

# Add a marker for the galactic center
#plt.plot(0, 0, 'r*', markersize=10, label='Galactic Center')
#plt.legend()

# Create a second plot with a minimum count threshold
plt.figure(figsize=(12, 10))

# Only show bins with at least 5 stars
min_count = 1
filtered_rotv = np.copy(mean_rotv)
filtered_rotv[count < min_count] = np.nan

mesh2 = plt.pcolormesh(xedges_kpc, yedges_kpc, filtered_rotv.T, 
                      shading='flat', cmap='viridis')
                      
# Add colorbar
cbar2 = plt.colorbar(mesh2)
cbar2.set_label('Mean Rotational Velocity (km/s)', size=12)

# Add labels and title
plt.xlabel('X (kpc)', fontsize=14)
plt.ylabel('Y (kpc)', fontsize=14)
plt.title(f'Binned Rotational Velocities (bins with ≥{min_count} stars)', fontsize=16)

# Set equal aspect ratio
plt.axis('equal')

# Add a marker for the galactic center
plt.plot(0, 0, 'r*', markersize=10, label='Galactic Center')
plt.legend()

# Save the filtered plot
plt.savefig('binned_rotational_velocity_map.png', dpi=300, bbox_inches='tight')

# Create a third plot with contours to highlight structures
plt.figure(figsize=(12, 10))

# Apply a Gaussian filter to smooth the data
from scipy.ndimage import gaussian_filter
smoothed_rotv = gaussian_filter(filtered_rotv, sigma=3)

# Plot the smoothed data
mesh3 = plt.pcolormesh(xedges_kpc, yedges_kpc, smoothed_rotv.T, 
                      shading='flat', cmap='viridis')

# Add contours
contour = plt.contour(
    (xedges_kpc[:-1] + xedges_kpc[1:]) / 2, 
    (yedges_kpc[:-1] + yedges_kpc[1:]) / 2, 
    smoothed_rotv.T, 
    levels=10, colors='white', alpha=0.5
)

# Add colorbar
cbar3 = plt.colorbar(mesh3)
cbar3.set_label('Smoothed Rotational Velocity (km/s)', size=12)

# Add labels and title
plt.xlabel('X (kpc)', fontsize=14)
plt.ylabel('Y (kpc)', fontsize=14)
plt.title('Smoothed Rotational Velocity Map with Contours', fontsize=16)

# Set equal aspect ratio
plt.axis('equal')

# Add a marker for the galactic center
plt.plot(0, 0, 'r*', markersize=10, label='Galactic Center')
plt.legend()

# Save the smoothed plot
plt.savefig('smoothed_rotational_velocity_map.png', dpi=300, bbox_inches='tight')

print("Analysis complete! Generated plots:")
print("1. Raw binned rotational velocities")
print("2. Filtered binned rotational velocities (saved as binned_rotational_velocity_map.png)")
print("3. Smoothed velocity map with contours (saved as smoothed_rotational_velocity_map.png)")

plt.show()

#%%

coeff=[0,
92.7597672696649,
-14.211008443582985,
0.9949471329796322,
-0.03127758534395031,
0.00031983624776082265]

coeff = coeff[::-1]

p1 = np.poly1d(coeff)

#print(p1(8.2))


#%%
# Calculate residual velocities (observed - model)

# Calculate radial distance of each bin center from galactic center (not from Sun)
xcenters = (xedges_kpc[:-1] + xedges_kpc[1:]) / 2
ycenters = (yedges_kpc[:-1] + yedges_kpc[1:]) / 2
X_centers, Y_centers = np.meshgrid(xcenters, ycenters)
# Transpose because X_centers and Y_centers have shape [y, x] from meshgrid
radial_distance = np.sqrt(X_centers.T**2 + Y_centers.T**2)

# Calculate expected rotational velocity at each bin's radial distance
expected_velocity = np.zeros_like(filtered_rotv)
for i in range(filtered_rotv.shape[0]):
    for j in range(filtered_rotv.shape[1]):
        if not np.isnan(filtered_rotv[i, j]):  # Only calculate for non-NaN values
            r = radial_distance[i, j]
            expected_velocity[i, j] = p1(abs(r))  # Use absolute value for radius
        else:
            expected_velocity[i, j] = np.nan  # Keep NaNs as NaNs

# Calculate residual velocity (observed - model)
# Only calculate for cells with valid data
residual_rotv = np.zeros_like(filtered_rotv)
residual_rotv[:] = np.nan  # Start with all NaNs
mask = ~np.isnan(filtered_rotv)  # Find valid data points
residual_rotv[mask] = filtered_rotv[mask] - expected_velocity[mask]

#filtering extreme data points
extfil=90
# Only calculate for cells with valid data
residual_rotv = np.zeros_like(filtered_rotv)
residual_rotv[:] = np.nan  # Start with all NaNs
mask = ~np.isnan(filtered_rotv)  # Find valid data points
residual_rotv[mask] = filtered_rotv[mask] - expected_velocity[mask]

# Set extreme residuals (>90 km/s) to 0
extreme_residual_mask = (np.abs(residual_rotv) > extfil) & (~np.isnan(residual_rotv))
if np.any(extreme_residual_mask):
    print(f"Found {np.sum(extreme_residual_mask)} extreme residuals (>90 km/s), setting to 0")
    residual_rotv[extreme_residual_mask] = 0

# Set up the residual plot
plt.figure(figsize=(12, 10))

# Create a diverging colormap centered at zero
from matplotlib.colors import CenteredNorm
norm = CenteredNorm(vcenter=0)

# Plot residual velocities - transpose for plotting
mesh_residual = plt.pcolormesh(xedges_kpc, yedges_kpc, residual_rotv.T, 
                              shading='flat', cmap='RdBu_r', norm=norm)
                      
# Add colorbar
cbar_residual = plt.colorbar(mesh_residual)
cbar_residual.set_label('Residual Velocity (km/s)', size=12)

# Add labels and title
plt.xlabel('X (kpc)', fontsize=14)
plt.ylabel('Y (kpc)', fontsize=14)
plt.title('Residual Rotational Velocity (Observed - Model)', fontsize=16)

# Set equal aspect ratio
plt.axis('equal')

# Add a marker for the galactic center
#plt.plot(0, 0, 'k*', markersize=10, label='Galactic Center')
plt.legend()

plt.savefig('residual_velocity_map.png', dpi=300, bbox_inches='tight')

# Create a smoothed residual plot
plt.figure(figsize=(12, 10))

# Apply a Gaussian filter to smooth the residuals
# First make a copy to avoid filtering the NaN values
smooth_data = np.copy(residual_rotv.T)
mask = ~np.isnan(smooth_data)

# Only smooth where we have data
from scipy.ndimage import gaussian_filter
smoothed_residual = np.zeros_like(smooth_data)
smoothed_residual[:] = np.nan

# Use a masked Gaussian filter to avoid edge effects from NaN values
valid_data = smooth_data[mask]
indices = np.array(np.where(mask)).T
if len(valid_data) > 0:
    filtered_valid = gaussian_filter(smooth_data[mask], sigma=1.0)
    for idx, val in zip(indices, filtered_valid):
        smoothed_residual[tuple(idx)] = val

# Plot smoothed residuals - already transposed
mesh_smooth_residual = plt.pcolormesh(xedges_kpc, yedges_kpc, smoothed_residual, 
                                     shading='flat', cmap='RdBu_r', norm=norm)

# Add contours only where data exists
contour_x = []
contour_y = []
contour_z = []
for i in range(len(xcenters)):
    for j in range(len(ycenters)):
        if not np.isnan(smoothed_residual[j, i]):
            contour_x.append(xcenters[i])
            contour_y.append(ycenters[j])
            contour_z.append(smoothed_residual[j, i])

if len(contour_z) > 10:  # Only add contours if we have enough points
    try:
        from scipy.interpolate import griddata
        xi = np.linspace(min(contour_x), max(contour_x), 100)
        yi = np.linspace(min(contour_y), max(contour_y), 100)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = griddata((contour_x, contour_y), contour_z, (Xi, Yi), method='cubic')
        
        contour_levels = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
        contour_residual = plt.contour(
            Xi, Yi, Zi, 
            levels=contour_levels, 
            colors='black', alpha=0.7,
            linewidths=0.5
        )
        plt.clabel(contour_residual, inline=True, fontsize=8, fmt='%1.0f')
    except:
        print("Could not generate contours - insufficient data points")

# Add colorbar
cbar_smooth_residual = plt.colorbar(mesh_smooth_residual)
cbar_smooth_residual.set_label('Smoothed Residual Velocity (km/s)', size=12)

# Add labels and title
plt.xlabel('X (kpc)', fontsize=14)
plt.ylabel('Y (kpc)', fontsize=14)
plt.title('Smoothed Residual Rotational Velocity', fontsize=16)

# Set equal aspect ratio
plt.axis('equal')

# Add a marker for the galactic center
#plt.plot(0, 0, 'k*', markersize=10, label='Galactic Center')
plt.legend()

# Save the smoothed residual plot
plt.savefig('smoothed_residual_velocity_map.png', dpi=300, bbox_inches='tight')

# Create a histogram of residuals
plt.figure(figsize=(10, 6))
valid_residuals = residual_rotv[~np.isnan(residual_rotv)]
if len(valid_residuals) > 0:
    plt.hist(valid_residuals.flatten(), bins=50, alpha=0.7, color='navy')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Residual Velocity (km/s)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Residual Velocities', fontsize=14)
    plt.grid(alpha=0.3)

    # Add statistics to the plot
    mean_residual = np.mean(valid_residuals)
    std_residual = np.std(valid_residuals)
    plt.text(0.05, 0.9, f'Mean: {mean_residual:.2f} km/s\nStd Dev: {std_residual:.2f} km/s', 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

    plt.savefig('residual_histogram.png', dpi=300, bbox_inches='tight')
else:
    print("Not enough data points for histogram")

print("\nResidual analysis complete!")
print("Generated additional plots:")
print("4. Residual velocity map (saved as residual_velocity_map.png)")
print("5. Smoothed residual velocity map (saved as smoothed_residual_velocity_map.png)")
print("6. Histogram of residual velocities (saved as residual_histogram.png)")

plt.show()



#%%
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

xedges_kpc=-xedges_kpc

# Solar neighborhood plot bounds
x_min, x_max = 10, 6.5
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
ax.plot(R_sun, 0, 'yo', label='Sun')



mesh_residual = plt.pcolormesh(xedges_kpc, yedges_kpc, residual_rotv.T, 
                              shading='flat', cmap='RdBu_r', norm=norm)








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
#%%

plt.figure(figsize=(12, 10))

# Plot residual velocities
mesh_residual = plt.pcolormesh(xedges_kpc, yedges_kpc, residual_rotv.T, 
                              shading='flat', cmap='RdBu_r', norm=norm)
                      
# Add colorbar
cbar_residual = plt.colorbar(mesh_residual)
cbar_residual.set_label('Residual Velocity (km/s)', size=12)

# Generate Local Spur for overlay
local_spur_params = spiral_arms["Local Spur"]
p_deg = local_spur_params["p"]
R_ref = local_spur_params["R_ref"]
theta_ref_deg = local_spur_params["theta_ref"]

# Convert to radians
p = np.radians(p_deg)
theta_ref = np.radians(theta_ref_deg)

# Generate spiral points
R_min = 7.8  # kpc, adjust as needed
R_max = 8.5  # kpc, adjust as needed
n_points = 1000
R_vals = np.linspace(R_min, R_max, n_points)
theta_vals = theta_ref + (1 / np.tan(p)) * np.log(R_vals / R_ref)
X_spur = R_vals * np.cos(theta_vals)
Y_spur = R_vals * np.sin(theta_vals)

# Plot Local Spur with thick white line for visibility
plt.plot(X_spur, Y_spur, 'w-', linewidth=3, label='Local Spur')
plt.plot(X_spur, Y_spur, 'k--', linewidth=1.5)

# Add Sun's position
plt.plot(-sun_x, sun_y, 'yo', markersize=10, label='Sun')

# Add labels and title
plt.xlabel('X (kpc)', fontsize=14)
plt.ylabel('Y (kpc)', fontsize=14)
plt.title('Residual Velocity with Local Spur Overlay', fontsize=16)

# Set equal aspect ratio
plt.axis('equal')

# Add legend
plt.legend(fontsize=12)

plt.show()