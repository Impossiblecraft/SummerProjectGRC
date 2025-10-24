"""
Copy Paste this code into uncert later after actually undertsanding how its is binning it

"""


#binned data (ignore below this point, it's for distance weighting)
# Add this section after your existing code

import scipy.stats as stats

# %%
# Distance-based weighting to correct for Gaia observational bias

def calculate_distance_weights(distances, weight_method='inverse_square'):
    """
    Calculate weights based on distance from Earth to correct for Gaia bias
    
    Parameters:
    -----------
    distances : array_like
        Distances from Earth in kpc
    weight_method : str
        Method for weighting: 'inverse_square', 'inverse', 'exponential', or 'linear'
        
    Returns:
    --------
    weights : array_like
        Weight factors for each star
    """
    
    if weight_method == 'inverse_square':
        # Weight by 1/d² to counteract volume effect
        weights = 1.0 / (distances**2 + 0.1)  # Add small value to avoid division by zero
        
    elif weight_method == 'inverse':
        # Weight by 1/d 
        weights = 1.0 / (distances + 0.1)
        
    elif weight_method == 'exponential':
        # Exponential weighting (more aggressive correction)
        scale_length = 2.0  # kpc
        weights = np.exp(distances / scale_length)
        
    elif weight_method == 'linear':
        # Linear increase with distance
        weights = distances / np.min(distances)
        
    else:
        raise ValueError("Unknown weight_method")
    
    # Normalize weights so they don't change overall statistics too much
    weights = weights / np.mean(weights)
    
    return weights

def bin_rotation_curve_distance_weighted(radii, velocities, radii_err, velocities_err, 
                                        distances, n_bins=15, min_points=5, 
                                        weight_method='inverse_square'):
    """
    Bin rotation curve data with distance-based weighting to correct for Gaia bias
    
    Parameters:
    -----------
    radii : array_like
        Galactocentric radial distances
    velocities : array_like  
        Rotational velocities
    radii_err, velocities_err : array_like
        Uncertainties in radii and velocities
    distances : array_like
        Distances from Earth (for weighting)
    n_bins : int
        Number of radial bins
    min_points : int
        Minimum number of stars required per bin
    weight_method : str
        Method for distance weighting
        
    Returns:
    --------
    dict with binned results
    """
    
    # Calculate distance weights
    distance_weights = calculate_distance_weights(distances, weight_method)
    
    # Define bin edges
    r_min, r_max = np.min(radii), np.max(radii)
    bin_edges = np.linspace(r_min, r_max, n_bins + 1)
    
    # Initialize arrays for results
    binned_velocities = np.full(n_bins, np.nan)
    binned_velocity_errors = np.full(n_bins, np.nan)
    binned_radii = np.full(n_bins, np.nan)
    binned_radii_errors = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)
    effective_counts = np.zeros(n_bins)  # Weighted counts
    
    for i in range(n_bins):
        # Find stars in this bin
        mask = (radii >= bin_edges[i]) & (radii < bin_edges[i+1])
        
        if np.sum(mask) >= min_points:
            # Get data for this bin
            bin_radii = radii[mask]
            bin_velocities = velocities[mask]
            bin_radii_err = radii_err[mask]
            bin_velocities_err = velocities_err[mask]
            bin_weights = distance_weights[mask]
            
            # Combined weights: distance weighting + measurement uncertainty weighting
            measurement_weights_vel = 1.0 / (bin_velocities_err**2 + 1e-10)
            measurement_weights_rad = 1.0 / (bin_radii_err**2 + 1e-10)
            
            # Combine distance weights with measurement precision weights
            combined_weights_vel = bin_weights * measurement_weights_vel
            combined_weights_rad = bin_weights * measurement_weights_rad
            
            # Weighted averages
            weighted_vel = np.sum(bin_velocities * combined_weights_vel) / np.sum(combined_weights_vel)
            weighted_rad = np.sum(bin_radii * combined_weights_rad) / np.sum(combined_weights_rad)
            
            # Weighted errors (accounting for both measurement and distance weighting)
            weighted_vel_err = np.sqrt(1.0 / np.sum(combined_weights_vel))
            weighted_rad_err = np.sqrt(1.0 / np.sum(combined_weights_rad))
            
            # Store results
            binned_velocities[i] = weighted_vel
            binned_velocity_errors[i] = weighted_vel_err
            binned_radii[i] = weighted_rad
            binned_radii_errors[i] = weighted_rad_err
            bin_counts[i] = np.sum(mask)
            effective_counts[i] = np.sum(bin_weights)  # Effective number of stars
    
    # Remove empty bins
    valid_bins = ~np.isnan(binned_velocities)
    
    return {
        'radii': binned_radii[valid_bins],
        'radii_err': binned_radii_errors[valid_bins],
        'velocities': binned_velocities[valid_bins],
        'velocities_err': binned_velocity_errors[valid_bins],
        'counts': bin_counts[valid_bins],
        'effective_counts': effective_counts[valid_bins]
    }

# %%
# Calculate distances from Earth for weighting
print("Calculating distance-based weights for Gaia bias correction...")

# Calculate distance from Earth (heliocentric distance)
earth_distances = np.sqrt((c[:, 0] - 8.2)**2 + c[:, 1]**2 + c[:, 2]**2)  # kpc from Earth
# Note: Sun is at approximately (8.2, 0, 0) in Galactocentric coordinates

# %%
# Compare different weighting schemes
weight_methods = ['inverse_square', 'inverse', 'exponential', 'linear']
colors = ['red', 'blue', 'green', 'orange']

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot distance distribution and weights
axes[0, 0].hist(earth_distances, bins=30, alpha=0.7, color='gray', 
                label=f'Distance distribution (n={len(earth_distances)})')
axes[0, 0].set_xlabel('Distance from Earth (kpc)')
axes[0, 0].set_ylabel('Number of stars')
axes[0, 0].set_title('Gaia Distance Distribution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot weights vs distance
for weight_method, color in zip(weight_methods, colors):
    weights = calculate_distance_weights(earth_distances, weight_method)
    
    # Sort for smooth plotting
    sort_idx = np.argsort(earth_distances)
    axes[0, 1].plot(earth_distances[sort_idx], weights[sort_idx], 
                    color=color, alpha=0.8, label=weight_method)

axes[0, 1].set_xlabel('Distance from Earth (kpc)')
axes[0, 1].set_ylabel('Weight Factor')
axes[0, 1].set_title('Distance Weighting Schemes')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Create binned curves with different weighting
binned_results = {}
for weight_method, color in zip(weight_methods, colors):
    binned_results[weight_method] = bin_rotation_curve_distance_weighted(
        d[:, 0], d[:, 1], d_err[:, 0], d_err[:, 1], 
        earth_distances, n_bins=15, weight_method=weight_method
    )
    
    axes[1, 0].errorbar(
        binned_results[weight_method]['radii'], 
        binned_results[weight_method]['velocities'],
        yerr=binned_results[weight_method]['velocities_err'],
        fmt='o-', capsize=3, markersize=4, alpha=0.8,
        color=color, label=f'{weight_method} weighting'
    )

# Add unweighted for comparison
unweighted_bins = bin_rotation_curve_distance_weighted(
    d[:, 0], d[:, 1], d_err[:, 0], d_err[:, 1], 
    earth_distances, weight_method='inverse_square'  # Will be overridden
)
# Actually create unweighted version
equal_weights = np.ones_like(earth_distances)
unweighted_mask = d[:, 0] > 0  # All valid data
r_bins = np.linspace(np.min(d[:, 0]), np.max(d[:, 0]), 16)
unweighted_binned_v = []
unweighted_binned_r = []
unweighted_errors = []

for i in range(len(r_bins)-1):
    mask = (d[:, 0] >= r_bins[i]) & (d[:, 0] < r_bins[i+1])
    if np.sum(mask) >= 5:
        unweighted_binned_v.append(np.mean(d[mask, 1]))
        unweighted_binned_r.append(np.mean(d[mask, 0]))
        unweighted_errors.append(np.std(d[mask, 1])/np.sqrt(np.sum(mask)))

axes[1, 0].errorbar(unweighted_binned_r, unweighted_binned_v, 
                    yerr=unweighted_errors, fmt='s-', capsize=3, 
                    markersize=4, color='black', alpha=0.8, label='No weighting')

axes[1, 0].set_xlabel('Galactocentric Radius (kpc)')
axes[1, 0].set_ylabel('Rotational Velocity (km/s)')
axes[1, 0].set_title('Distance-Weighted Rotation Curves')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot effective vs actual counts
best_method = 'inverse_square'  # Choose your preferred method
axes[1, 1].bar(binned_results[best_method]['radii'] - 0.2, 
               binned_results[best_method]['counts'], 
               width=0.4, alpha=0.7, color='blue', 
               label='Actual star count')
axes[1, 1].bar(binned_results[best_method]['radii'] + 0.2, 
               binned_results[best_method]['effective_counts'], 
               width=0.4, alpha=0.7, color='red', 
               label='Effective weighted count')
axes[1, 1].set_xlabel('Galactocentric Radius (kpc)')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title(f'Bin Populations ({best_method} weighting)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Save distance-weighted rotation curve (using inverse_square as recommended)
distance_weighted_df = pd.DataFrame({
    'radius_kpc': binned_results['inverse_square']['radii'],
    'radius_err_kpc': binned_results['inverse_square']['radii_err'],
    'rot_velocity_kms': binned_results['inverse_square']['velocities'],
    'rot_velocity_err_kms': binned_results['inverse_square']['velocities_err'],
    'star_count': binned_results['inverse_square']['counts'],
    'effective_count': binned_results['inverse_square']['effective_counts']
})

distance_weighted_df.to_csv("distance_weighted_rotation_curve.csv", index=False)

# %%
#Binned data
# Print comparison statistics
print("\n=== Distance Weighting Analysis ===")
print(f"Distance range: {np.min(earth_distances):.2f} - {np.max(earth_distances):.2f} kpc")
print(f"Median distance: {np.median(earth_distances):.2f} kpc")

print(f"\nStars within 1 kpc of Sun: {np.sum(earth_distances < 1.0)}")
print(f"Stars beyond 5 kpc from Sun: {np.sum(earth_distances > 5.0)}")
print(f"Ratio (far/near): {np.sum(earth_distances > 5.0) / np.sum(earth_distances < 1.0):.2f}")

for method in weight_methods:
    weights = calculate_distance_weights(earth_distances, method)
    print(f"\n{method} weighting:")
    print(f"  Weight range: {np.min(weights):.2f} - {np.max(weights):.2f}")
    print(f"  Distant stars (>5kpc) weight boost: {np.mean(weights[earth_distances > 5.0]):.2f}x")

# %%
# Diagnostic plot: Show weight effect across the galaxy
plt.figure(figsize=(12, 5))

# Plot 1: Spatial distribution with weights
plt.subplot(1, 2, 1)
weights_inv_sq = calculate_distance_weights(earth_distances, 'inverse_square')
scatter = plt.scatter(c[:, 0], c[:, 1], c=weights_inv_sq, s=2, 
                     cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Distance Weight')
plt.xlabel('Galactocentric X (kpc)')
plt.ylabel('Galactocentric Y (kpc)')
plt.title('Spatial Distribution of Distance Weights')
plt.axis('equal')

# Plot 2: Weight vs galactocentric radius
plt.subplot(1, 2, 2)
plt.scatter(d[:, 0], weights_inv_sq, alpha=0.5, s=1)
plt.xlabel('Galactocentric Radius (kpc)')
plt.ylabel('Distance Weight')
plt.title('Distance Weight vs Galactocentric Radius')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nSaved distance-weighted rotation curve with {len(distance_weighted_df)} data points")
print("Recommended: Use 'inverse_square' weighting to best correct for Gaia observational bias")