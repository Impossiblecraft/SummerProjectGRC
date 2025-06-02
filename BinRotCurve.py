import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

# Load data with uncertainties from UncertProp
data = pd.read_csv("rotcurvebin80k1.csv")

radii = data['pos_kpc'].values
velocities = data['rotv_kms'].values
radii_err = data['pos_err_kpc'].values
velocities_err = data['rotv_err_kms'].values

def calculate_fractional_contributions(radii, radii_err, bin_start, bin_end, n_bins=15):
    """
    Calculate fractional contribution of each data point to each bin
    using renormalized truncated Gaussian based on radial uncertainties.
    
    Parameters:
    -----------
    radii : array_like
        Radial distances for each star
    radii_err : array_like  
        Uncertainties in radial distances (xerr)
    n_bins : int
        Number of bins
        
    Returns:
    --------
    contributions : 2D array
        Shape (n_stars, n_bins) - fractional contribution of each star to each bin
    bin_edges : array
        Bin edge positions
    bin_centers : array
        Bin center positions
    """
    
    n_stars = len(radii)
    
    # Define bin edges
    #r_min, r_max = np.min(radii), np.max(radii)
    r_min=bin_start
    r_max=bin_end
    bin_edges = np.linspace(r_min, r_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Initialize contribution matrix
    contributions = np.zeros((n_stars, n_bins))
    
    # For each star, calculate its contribution to each bin
    for i, (r, r_err) in enumerate(zip(radii, radii_err)):
        
        # Define truncation bounds (typically ±3σ or based on your preference)
        sigma = r_err
        lower_bound = r - 3 * sigma  # Lower truncation
        upper_bound = r + 3 * sigma  # Upper truncation
        
        # Create truncated normal distribution
        # truncnorm uses standardized bounds: (a, b) where a,b = (bound - mean)/sigma
        a = (lower_bound - r) / sigma  # Standardized lower bound
        b = (upper_bound - r) / sigma  # Standardized upper bound
        
        # For each bin, calculate the probability mass within that bin
        for j in range(n_bins):
            bin_left = bin_edges[j]
            bin_right = bin_edges[j + 1]
            
            # Convert bin edges to standardized coordinates
            bin_left_std = (bin_left - r) / sigma
            bin_right_std = (bin_right - r) / sigma
            
            # Calculate probability mass in this bin using truncated normal CDF
            # P(bin_left < X < bin_right | truncated normal)
            prob_left = truncnorm.cdf(bin_left_std, a, b)
            prob_right = truncnorm.cdf(bin_right_std, a, b)
            
            # Fractional contribution is the probability mass in this bin
            contributions[i, j] = prob_right - prob_left
    
    return contributions, bin_edges, bin_centers

# %%
# Calculate fractional contributions
print("Calculating fractional contributions using truncated Gaussian...")

contributions, bin_edges, bin_centers = calculate_fractional_contributions(
    radii, radii_err, np.min(radii), np.max(radii),  n_bins=15
)

print(f"Contribution matrix shape: {contributions.shape}")
print(f"Sum of contributions per star (should be ~1.0): {np.sum(contributions, axis=1)[:5]}")
"""
# %%
# Visualize the contribution matrix
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Contribution matrix as heatmap
im1 = axes[0, 0].imshow(contributions.T, aspect='auto', cmap='viridis', 
                        extent=[0, len(radii), bin_centers[0], bin_centers[-1]])
axes[0, 0].set_xlabel('Star Index')
axes[0, 0].set_ylabel('Bin Center (kpc)')
axes[0, 0].set_title('Fractional Contributions Matrix')
plt.colorbar(im1, ax=axes[0, 0], label='Contribution Fraction')

# Plot 2: Example contributions for a few stars
star_indices = [0, len(radii)//4, len(radii)//2, 3*len(radii)//4, len(radii)-1]
for idx in star_indices:
    axes[0, 1].plot(bin_centers, contributions[idx, :], 'o-', 
                    label=f'Star {idx}: r={radii[idx]:.2f}±{radii_err[idx]:.2f}', alpha=0.8)
axes[0, 1].set_xlabel('Bin Center (kpc)')
axes[0, 1].set_ylabel('Contribution Fraction')
axes[0, 1].set_title('Example Star Contributions to Bins')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Total contributions per bin (sum over all stars)
total_contributions_per_bin = np.sum(contributions, axis=0)
axes[1, 0].bar(bin_centers, total_contributions_per_bin, 
               width=np.diff(bin_centers)[0]*0.8, alpha=0.7, color='blue')
axes[1, 0].set_xlabel('Bin Center (kpc)')
axes[1, 0].set_ylabel('Total Fractional Contributions')
axes[1, 0].set_title('Sum of All Star Contributions per Bin')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Distribution of number of bins each star contributes to significantly
significant_contributions = contributions > 0.01  # More than 1% contribution
bins_per_star = np.sum(significant_contributions, axis=1)
axes[1, 1].hist(bins_per_star, bins=range(1, np.max(bins_per_star)+2), 
                alpha=0.7, color='green', edgecolor='black')
axes[1, 1].set_xlabel('Number of Bins with Significant Contribution')
axes[1, 1].set_ylabel('Number of Stars')
axes[1, 1].set_title('Distribution of Bin Contributions per Star')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Save the contribution matrix and bin information
np.save("fractional_contributions_matrix.npy", contributions)
np.save("bin_centers.npy", bin_centers)
np.save("bin_edges.npy", bin_edges)

# Also save as CSV for easy inspection
contributions_df = pd.DataFrame(contributions, 
                               columns=[f'bin_{i}' for i in range(len(bin_centers))])
contributions_df['star_radius'] = radii
contributions_df['star_radius_err'] = radii_err
contributions_df['star_velocity'] = velocities
contributions_df['star_velocity_err'] = velocities_err

contributions_df.to_csv("fractional_contributions.csv", index=False)

print("\nSaved fractional contributions:")
print("- fractional_contributions_matrix.npy (NumPy array)")
print("- fractional_contributions.csv (CSV with star data)")
print("- bin_centers.npy and bin_edges.npy")

# %%
# Print summary statistics
print(f"\n=== Fractional Contribution Statistics ===")
print(f"Number of stars: {contributions.shape[0]}")
print(f"Number of bins: {contributions.shape[1]}")
print(f"Average contribution per star per bin: {np.mean(contributions):.4f}")
print(f"Maximum single contribution: {np.max(contributions):.4f}")
print(f"Average number of bins per star (>1% contribution): {np.mean(bins_per_star):.1f}")
print(f"Stars contributing to only 1 bin: {np.sum(bins_per_star == 1)}")
print(f"Stars contributing to >5 bins: {np.sum(bins_per_star > 5)}")

# Verify normalization
row_sums = np.sum(contributions, axis=1)
print(f"Row sum statistics (should be ~1.0):")
print(f"  Mean: {np.mean(row_sums):.6f}")
print(f"  Std:  {np.std(row_sums):.6f}")
print(f"  Min:  {np.min(row_sums):.6f}")
print(f"  Max:  {np.max(row_sums):.6f}")
"""

#%%

#Implementing y-err calcs

def calculate_binned_velocities_and_errors(velocities, velocities_err, contributions):
    """
    Calculate binned velocities and their errors using fractional contributions
    
    Parameters:
    -----------
    velocities : array_like
        Rotational velocities for each star
    velocities_err : array_like
        Uncertainties in velocities for each star
    contributions : 2D array
        Shape (n_stars, n_bins) - fractional contribution matrix
        
    Returns:
    --------
    binned_velocities : array
        Weighted mean velocity for each bin
    binned_velocity_errors : array
        Weighted mean of velocity errors for each bin
    effective_counts : array
        Sum of contributions per bin (effective number of stars)
    """
    
    n_stars, n_bins = contributions.shape
    
    # Initialize output arrays
    binned_velocities = np.zeros(n_bins)
    binned_velocity_errors = np.zeros(n_bins)
    effective_counts = np.zeros(n_bins)
    
    for j in range(n_bins):
        # Get contributions for this bin
        bin_contributions = contributions[:, j]
        
        # Only consider stars that contribute significantly to this bin
        contributing_mask = bin_contributions > 1e-10  # Avoid numerical issues
        
        if np.sum(contributing_mask) > 0:
            # Extract data for contributing stars
            contributing_velocities = velocities[contributing_mask]
            contributing_velocity_errors = velocities_err[contributing_mask]
            contributing_weights = bin_contributions[contributing_mask]
            
            # Calculate weighted mean velocity
            weighted_velocity = np.sum(contributing_velocities * contributing_weights) / np.sum(contributing_weights)
            
            # Calculate weighted mean of velocity errors
            weighted_velocity_error = np.sum(contributing_velocity_errors * contributing_weights) / np.sum(contributing_weights)
            
            # Store results
            binned_velocities[j] = weighted_velocity
            binned_velocity_errors[j] = weighted_velocity_error
            effective_counts[j] = np.sum(contributing_weights)
        else:
            # No significant contributions to this bin
            binned_velocities[j] = np.nan
            binned_velocity_errors[j] = np.nan
            effective_counts[j] = 0
    
    return binned_velocities, binned_velocity_errors, effective_counts

# Calculate binned velocities and errors
print("Calculating binned velocities and errors using contribution matrix...")

binned_velocities, binned_velocity_errors, effective_counts = calculate_binned_velocities_and_errors(
    velocities, velocities_err, contributions
)

# Remove bins with no contributions
valid_bins = ~np.isnan(binned_velocities)
final_bin_centers = bin_centers[valid_bins]
final_binned_velocities = binned_velocities[valid_bins]
final_binned_velocity_errors = binned_velocity_errors[valid_bins]
final_effective_counts = effective_counts[valid_bins]

print(f"Number of valid bins: {len(final_bin_centers)}")
print(f"Effective counts per bin: {final_effective_counts}")

#%%

# Visualize the binned velocities and errors

# ...existing code...

#%%
# Visualize the binned velocities and errors
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Original data
axes[0, 0].errorbar(radii, velocities, xerr=radii_err, yerr=velocities_err,
                    fmt='.', alpha=0.3, markersize=1, capsize=0)
axes[0, 0].set_xlabel('Radial Distance (kpc)')
axes[0, 0].set_ylabel('Rotational Velocity (km/s)')
axes[0, 0].set_title('Original Data with Uncertainties')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Fractionally binned data
axes[0, 1].errorbar(final_bin_centers, final_binned_velocities, 
                    yerr=final_binned_velocity_errors,
                    fmt='o-', capsize=5, markersize=8, linewidth=2, color='red')
axes[0, 1].set_xlabel('Radial Distance (kpc)')
axes[0, 1].set_ylabel('Rotational Velocity (km/s)')
axes[0, 1].set_title('Fractionally Binned Rotation Curve')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Comparison overlay
axes[1, 0].scatter(radii, velocities, alpha=0.2, s=1, color='gray', label='Individual stars')
axes[1, 0].errorbar(final_bin_centers, final_binned_velocities, 
                    yerr=final_binned_velocity_errors,
                    fmt='o-', capsize=5, markersize=8, linewidth=3, 
                    color='red', label='Fractional binning')
axes[1, 0].set_xlabel('Radial Distance (kpc)')
axes[1, 0].set_ylabel('Rotational Velocity (km/s)')
axes[1, 0].set_title('Fractional Binning vs Original Data')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Effective counts per bin
axes[1, 1].bar(final_bin_centers, final_effective_counts,
               width=np.diff(final_bin_centers)[0]*0.8 if len(final_bin_centers) > 1 else 1.0,
               alpha=0.7, color='blue', edgecolor='darkblue')
axes[1, 1].set_xlabel('Radial Distance (kpc)')
axes[1, 1].set_ylabel('Effective Star Count')
axes[1, 1].set_title('Effective Number of Stars per Bin')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%%
# Save the final binned rotation curve
binned_rotation_curve = pd.DataFrame({
    'radius_kpc': final_bin_centers,
    'velocity_kms': final_binned_velocities,
    'velocity_err_kms': final_binned_velocity_errors,
    'effective_star_count': final_effective_counts
})

binned_rotation_curve.to_csv("fractionally_binned_rotation_curve.csv", index=False)
print("Saved fractionally binned rotation curve to: fractionally_binned_rotation_curve.csv")

#%%
# Print summary statistics
print(f"\n=== Fractionally Binned Rotation Curve Statistics ===")
print(f"Number of original stars: {len(velocities)}")
print(f"Number of valid bins: {len(final_bin_centers)}")
print(f"Velocity range: {np.min(final_binned_velocities):.1f} - {np.max(final_binned_velocities):.1f} km/s")
print(f"Mean velocity error: {np.mean(final_binned_velocity_errors):.2f} km/s")
print(f"Average effective stars per bin: {np.mean(final_effective_counts):.1f}")
print(f"Min/Max effective counts: {np.min(final_effective_counts):.1f} / {np.max(final_effective_counts):.1f}")

# Compare error reduction
original_mean_error = np.mean(velocities_err)
binned_mean_error = np.mean(final_binned_velocity_errors)
error_reduction_factor = original_mean_error / binned_mean_error

print(f"\nError Analysis:")
print(f"Original mean velocity error: {original_mean_error:.2f} km/s")
print(f"Binned mean velocity error: {binned_mean_error:.2f} km/s")
print(f"Error reduction factor: {error_reduction_factor:.1f}x")




