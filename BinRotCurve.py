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
    radii, radii_err, np.min(radii), np.max(radii),  n_bins=30 #15
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
#Ignore code below (commneted out for a reason)
"""
# Visualize the binned velocities and errors

# ...existing code...
"""

#%%
#vel second implementation
# ...existing code...
#measerr=np.zeros(15)
#scterr=np.zeros(15)

def calculate_binned_velocities_and_errors(velocities, velocities_err, contributions):
    """
    Calculate binned velocities and their errors using fractional contributions
    Combines measurement error and scatter error intelligently
    
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
        Combined error (measurement + scatter)
    effective_counts : array
        Sum of contributions per bin (effective number of stars)
    """
    
    n_stars, n_bins = contributions.shape
    
    # Initialize output arrays
    binned_velocities = np.zeros(n_bins)
    binned_velocity_errors = np.zeros(n_bins)
    effective_counts = np.zeros(n_bins)
    
    # Also track components for analysis
    measurement_errors = np.zeros(n_bins)
    scatter_errors = np.zeros(n_bins)
    
    for j in range(n_bins):
        # Get contributions for this bin
        bin_contributions = contributions[:, j]
        
        # Only consider stars that contribute significantly to this bin
        contributing_mask = bin_contributions > 1e-10
        
        if np.sum(contributing_mask) > 0:
            # Extract data for contributing stars
            contributing_velocities = velocities[contributing_mask]
            contributing_velocity_errors = velocities_err[contributing_mask]
            contributing_weights = bin_contributions[contributing_mask]
            
            # Calculate weighted mean velocity
            weighted_velocity = np.sum(contributing_velocities * contributing_weights) / np.sum(contributing_weights)
            
            # 1. Measurement error: weighted mean of individual measurement uncertainties
            measurement_error = np.sum(contributing_velocity_errors * contributing_weights) / np.sum(contributing_weights)
            
            # 2. Scatter error: standard error of the weighted mean from velocity scatter
            velocity_deviations = contributing_velocities - weighted_velocity
            weighted_variance = np.sum(contributing_weights * velocity_deviations**2) / np.sum(contributing_weights)
            velocity_scatter = np.sqrt(weighted_variance)
            
            # Standard error of weighted mean
            effective_sample_size = (np.sum(contributing_weights))**2 / np.sum(contributing_weights**2)
            scatter_error = velocity_scatter / np.sqrt(effective_sample_size) 
            
            #Should we divide by root N or not (is it appropriiate for data since it isn't measurment of a single random variable)
            
            # 3. Combine errors intelligently
            # If similar magnitude (within factor of 2), use the larger one
            # Otherwise, add in quadrature
            ratio = max(measurement_error, scatter_error) / min(measurement_error, scatter_error)
            
            if ratio <= 2:  # Similar values - use the larger one
                combined_error = max(measurement_error, scatter_error)
            else:  # Different magnitudes - add in quadrature
                combined_error = np.sqrt(measurement_error**2 + scatter_error**2)
            
            # Store results
            binned_velocities[j] = weighted_velocity
            binned_velocity_errors[j] = combined_error
            measurement_errors[j] = measurement_error
            scatter_errors[j] = scatter_error
            effective_counts[j] = np.sum(contributing_weights)
            
        else:
            # No significant contributions to this bin
            binned_velocities[j] = np.nan
            binned_velocity_errors[j] = np.nan
            measurement_errors[j] = np.nan
            scatter_errors[j] = np.nan
            effective_counts[j] = 0
    
    # Store error components for analysis
    #measerr = measurement_errors
    #scterr = scatter_errors
    
    return binned_velocities, binned_velocity_errors, effective_counts, measurement_errors, scatter_errors

# Calculate binned velocities and errors with intelligent error combination
print("Calculating binned velocities and errors (measurement + scatter)...")

binned_velocities, binned_velocity_errors, effective_counts, measerr, scterr = calculate_binned_velocities_and_errors(
    velocities, velocities_err, contributions
)

# Remove bins with no contributions
valid_bins = ~np.isnan(binned_velocities)
final_bin_centers = bin_centers[valid_bins]
final_binned_velocities = binned_velocities[valid_bins]
final_binned_velocity_errors = binned_velocity_errors[valid_bins]
final_effective_counts = effective_counts[valid_bins]

# Extract error components for analysis
final_measurement_errors = measerr[valid_bins]
final_scatter_errors = scterr[valid_bins]

print(f"Number of valid bins: {len(final_bin_centers)}")
print(f"Effective counts per bin: {final_effective_counts}")

#%%
# Add error analysis plot
plt.figure(figsize=(15, 10))

# Plot 1: Error components comparison
plt.subplot(2, 3, 1)
plt.plot(final_bin_centers, final_measurement_errors, 'bo-', label='Measurement error', markersize=6)
plt.plot(final_bin_centers, final_scatter_errors, 'ro-', label='Scatter error', markersize=6)
plt.plot(final_bin_centers, final_binned_velocity_errors, 'ko-', label='Combined error', markersize=6)
plt.xlabel('Radial Distance (kpc)')
plt.ylabel('Velocity Error (km/s)')
plt.title('Error Components vs Radius')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Error ratio and combination method
plt.subplot(2, 3, 2)
error_ratios = np.maximum(final_measurement_errors, final_scatter_errors) / np.minimum(final_measurement_errors, final_scatter_errors)
combination_method = ['Max' if ratio <= 2.0 else 'Quadrature' for ratio in error_ratios]
colors = ['blue' if method == 'Max' else 'red' for method in combination_method]

plt.scatter(final_bin_centers, error_ratios, c=colors, s=50)
plt.axhline(y=2.0, color='black', linestyle='--', alpha=0.7, label='Threshold (ratio=2)')
plt.xlabel('Radial Distance (kpc)')
plt.ylabel('Error Ratio (larger/smaller)')
plt.title('Error Combination Method Used')
plt.legend(['Quadrature', 'Maximum', 'Threshold'])
plt.grid(True, alpha=0.3)

# Plot 3: Fractionally binned rotation curve
plt.subplot(2, 3, 3)
plt.errorbar(final_bin_centers, final_binned_velocities, 
             yerr=final_binned_velocity_errors,
             fmt='o-', capsize=5, markersize=8, linewidth=2, color='red')
plt.scatter(radii, velocities, alpha=0.1, s=1, color='gray')
plt.xlabel('Radial Distance (kpc)')
plt.ylabel('Rotational Velocity (km/s)')
plt.title('Binned Rotation Curve')
plt.grid(True, alpha=0.3)

# Plot 4: Original data
plt.subplot(2, 3, 4)
plt.errorbar(radii, velocities, xerr=radii_err, yerr=velocities_err,
             fmt='.', alpha=0.3, markersize=1, capsize=0)
plt.xlabel('Radial Distance (kpc)')
plt.ylabel('Rotational Velocity (km/s)')
plt.title('Original Data with Uncertainties')
plt.grid(True, alpha=0.3)

# Plot 5: Comparison overlay
plt.subplot(2, 3, 5)
plt.scatter(radii, velocities, alpha=0.2, s=1, color='gray', label='Individual stars')
plt.errorbar(final_bin_centers, final_binned_velocities, 
             yerr=final_binned_velocity_errors,
             fmt='o-', capsize=5, markersize=8, linewidth=3, 
             color='red', label='Fractional binning')
plt.xlabel('Radial Distance (kpc)')
plt.ylabel('Rotational Velocity (km/s)')
plt.title('Binning vs Original Data')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Effective counts per bin
plt.subplot(2, 3, 6)
plt.bar(final_bin_centers, final_effective_counts,
        width=np.diff(final_bin_centers)[0]*0.8 if len(final_bin_centers) > 1 else 1.0,
        alpha=0.7, color='blue', edgecolor='darkblue')
plt.xlabel('Radial Distance (kpc)')
plt.ylabel('Effective Star Count')
plt.title('Effective Stars per Bin')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%%
# Print detailed error analysis
print(f"\n=== Error Analysis Summary ===")
print(f"Average measurement error: {np.mean(final_measurement_errors):.2f} km/s")
print(f"Average scatter error: {np.mean(final_scatter_errors):.2f} km/s")
print(f"Average combined error: {np.mean(final_binned_velocity_errors):.2f} km/s")

# Count combination methods used
max_used = sum(1 for ratio in error_ratios if ratio <= 2.0)
quad_used = len(error_ratios) - max_used

print(f"\nCombination methods used:")
print(f"  Maximum method: {max_used} bins ({100*max_used/len(error_ratios):.1f}%)")
print(f"  Quadrature method: {quad_used} bins ({100*quad_used/len(error_ratios):.1f}%)")

print(f"\nBins where measurement error dominates: {sum(1 for i in range(len(final_measurement_errors)) if final_measurement_errors[i] > final_scatter_errors[i])}")
print(f"Bins where scatter error dominates: {sum(1 for i in range(len(final_measurement_errors)) if final_scatter_errors[i] > final_measurement_errors[i])}")

#%%
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.errorbar(radii, velocities, xerr=radii_err, yerr=velocities_err,
             fmt='.', alpha=0.3, markersize=1, capsize=0)
plt.xlabel('Radial Distance (kpc)')
plt.ylabel('Rotational Velocity (km/s)')
plt.title('Original Data with Uncertainties')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(radii, velocities, alpha=0.2, s=1, color='gray', label='Individual stars')
plt.errorbar(final_bin_centers, final_binned_velocities, 
             yerr=final_binned_velocity_errors,
             fmt='o-', capsize=5, markersize=8, linewidth=3, 
             color='red', label='Fractional binning')
plt.xlabel('Radial Distance (kpc)')
plt.ylabel('Rotational Velocity (km/s)')
plt.title('Binning vs Original Data')
plt.legend()
plt.grid(True, alpha=0.3)


plt.tight_layout()
#plt.savefig("BinnedTemp1.png")
plt.show()

print(final_effective_counts)


#Decide whether to dividie by root N'

#Implement Jean's Equation



#%%
# Curve fitting with power series

from scipy.optimize import curve_fit
from scipy.special import factorial

def power_series_fit(r, *coeffs):
    """
    Power series function: v(r) = a0 + a1*r + a2*r^2 + a3*r^3 + ...
    
    Parameters:
    -----------
    r : array_like
        Radial distances
    *coeffs : tuple
        Coefficients of the power series (a0, a1, a2, ...)
    
    Returns:
    --------
    v : array_like
        Velocity values at radii r
    """
    result = np.zeros_like(r)
    for i, coeff in enumerate(coeffs):
        result += coeff * r**i
    return result

def fit_power_series(radii, velocities, velocity_errors, max_degree=4):
    """
    Fit a power series to rotation curve data
    
    Parameters:
    -----------
    radii : array_like
        Radial distances of bins
    velocities : array_like
        Velocity values of bins
    velocity_errors : array_like
        Uncertainty in velocities
    max_degree : int
        Maximum degree of polynomial to fit
        
    Returns:
    --------
    dict with fitting results for different degrees
    """
    
    results = {}
    
    for degree in range(1, max_degree + 1):
        try:
            # Define the function for this degree
            def poly_func(r, *coeffs):
                return power_series_fit(r, *coeffs)
            
            # Initial guess (all coefficients = 1)
            initial_guess = np.ones(degree + 1)
            
            # Fit the curve with error weighting
            popt, pcov = curve_fit(
                poly_func, radii, velocities, 
                sigma=velocity_errors, 
                p0=initial_guess,
                absolute_sigma=True,
                maxfev=5000
            )
            
            # Calculate R-squared and reduced chi-squared
            y_pred = power_series_fit(radii, *popt)
            ss_res = np.sum((velocities - y_pred)**2)
            ss_tot = np.sum((velocities - np.mean(velocities))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Reduced chi-squared
            chi_squared = np.sum(((velocities - y_pred) / velocity_errors)**2)
            reduced_chi_squared = chi_squared / (len(velocities) - len(popt))
            
            # Calculate parameter uncertainties
            param_errors = np.sqrt(np.diag(pcov))
            
            # Store results
            results[degree] = {
                'coefficients': popt,
                'covariance': pcov,
                'parameter_errors': param_errors,
                'r_squared': r_squared,
                'reduced_chi_squared': reduced_chi_squared,
                'aic': len(velocities) * np.log(ss_res/len(velocities)) + 2*len(popt),  # AIC
                'bic': len(velocities) * np.log(ss_res/len(velocities)) + len(popt)*np.log(len(velocities))  # BIC
            }
            
        except Exception as e:
            print(f"Failed to fit degree {degree} polynomial: {e}")
            results[degree] = None
    
    return results

# Perform power series fitting
print("Fitting power series to binned rotation curve...")

fit_results = fit_power_series(final_bin_centers, final_binned_velocities, 
                              final_binned_velocity_errors, max_degree=5)

# Print fitting results
print(f"\n=== Power Series Fitting Results ===")
for degree, result in fit_results.items():
    if result is not None:
        print(f"\nDegree {degree} polynomial:")
        print(f"  R²: {result['r_squared']:.4f}")
        print(f"  Reduced χ²: {result['reduced_chi_squared']:.4f}")
        print(f"  AIC: {result['aic']:.2f}")
        print(f"  BIC: {result['bic']:.2f}")
        print("  Coefficients:")
        for i, (coeff, error) in enumerate(zip(result['coefficients'], result['parameter_errors'])):
            print(f"    a{i}: {coeff:.4f} ± {error:.4f}")

# Find the best fit based on multiple criteria
def select_best_fit(fit_results):
    """Select best fit based on multiple criteria"""
    valid_results = {k: v for k, v in fit_results.items() if v is not None}
    
    if not valid_results:
        return None
    
    best_degree = None
    best_score = float('inf')
    
    for degree, result in valid_results.items():
        # Weighted score combining multiple criteria
        # Lower AIC/BIC is better, reduced chi-squared should be close to 1
        score = (result['aic'] + result['bic'] + 
                10 * abs(result['reduced_chi_squared'] - 1))
        
        if score < best_score:
            best_score = score
            best_degree = degree
    
    return best_degree

best_degree = select_best_fit(fit_results)
print(f"\nBest fit: Degree {best_degree} polynomial")



#%%
# Visualization of fits
plt.figure(figsize=(18, 12))

# Plot 1: All fits overlaid
plt.subplot(2, 3, 1)
plt.errorbar(final_bin_centers, final_binned_velocities, 
             yerr=final_binned_velocity_errors,
             fmt='ko', capsize=5, markersize=6, label='Binned data')

# Create fine grid for smooth curves
r_fine = np.linspace(final_bin_centers.min(), final_bin_centers.max(), 200)
colors = ['red', 'blue', 'green', 'orange', 'purple']

for i, (degree, result) in enumerate(fit_results.items()):
    if result is not None and degree <= len(colors):
        v_fine = power_series_fit(r_fine, *result['coefficients'])
        plt.plot(r_fine, v_fine, colors[i], linewidth=2, 
                label=f'Degree {degree} (R²={result["r_squared"]:.3f})')

plt.xlabel('Radial Distance (kpc)')
plt.ylabel('Rotational Velocity (km/s)')
plt.title('Power Series Fits')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Best fit with residuals
plt.subplot(2, 3, 2)
plt.errorbar(final_bin_centers, final_binned_velocities, 
             yerr=final_binned_velocity_errors,
             fmt='ko', capsize=5, markersize=8, label='Binned data')
plt.scatter(radii, velocities, alpha=0.2, s=1, color='gray', label='Individual stars')

if best_degree is not None:
    best_result = fit_results[best_degree]
    v_fine = power_series_fit(r_fine, *best_result['coefficients'])
    v_fit = power_series_fit(final_bin_centers, *best_result['coefficients'])
    
    plt.plot(r_fine, v_fine, 'red', linewidth=3, 
            label=f'Best fit (degree {best_degree})')
    
plt.xlabel('Radial Distance (kpc)')
plt.ylabel('Rotational Velocity (km/s)')
plt.title('Best Power Series Fit')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Residuals
plt.subplot(2, 3, 3)
if best_degree is not None:
    residuals = final_binned_velocities - v_fit
    plt.errorbar(final_bin_centers, residuals, 
                yerr=final_binned_velocity_errors,
                fmt='ro', capsize=5, markersize=6)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    
plt.xlabel('Radial Distance (kpc)')
plt.ylabel('Residuals (km/s)')
plt.title('Fit Residuals')
plt.grid(True, alpha=0.3)

# Plot 4: Model selection criteria
plt.subplot(2, 3, 4)
degrees = []
aics = []
bics = []
for degree, result in fit_results.items():
    if result is not None:
        degrees.append(degree)
        aics.append(result['aic'])
        bics.append(result['bic'])

plt.plot(degrees, aics, 'bo-', label='AIC', markersize=8)
plt.plot(degrees, bics, 'ro-', label='BIC', markersize=8)
plt.xlabel('Polynomial Degree')
plt.ylabel('Information Criterion')
plt.title('Model Selection Criteria')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: R-squared and reduced chi-squared
plt.subplot(2, 3, 5)
r_squareds = []
red_chi_squareds = []
for degree, result in fit_results.items():
    if result is not None:
        r_squareds.append(result['r_squared'])
        red_chi_squareds.append(result['reduced_chi_squared'])

plt.plot(degrees, r_squareds, 'go-', label='R²', markersize=8)
plt.plot(degrees, red_chi_squareds, 'mo-', label='Reduced χ²', markersize=8)
plt.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='χ² = 1')
plt.xlabel('Polynomial Degree')
plt.ylabel('Goodness of Fit')
plt.title('Fit Quality Metrics')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Coefficient evolution
plt.subplot(2, 3, 6)
max_coeffs = max(len(result['coefficients']) for result in fit_results.values() if result is not None)
coeff_colors = plt.cm.viridis(np.linspace(0, 1, max_coeffs))

for coeff_idx in range(max_coeffs):
    coeff_values = []
    coeff_errors = []
    valid_degrees = []
    
    for degree, result in fit_results.items():
        if result is not None and coeff_idx < len(result['coefficients']):
            valid_degrees.append(degree)
            coeff_values.append(result['coefficients'][coeff_idx])
            coeff_errors.append(result['parameter_errors'][coeff_idx])
    
    if valid_degrees:
        plt.errorbar(valid_degrees, coeff_values, yerr=coeff_errors,
                    fmt='o-', color=coeff_colors[coeff_idx], 
                    label=f'a{coeff_idx}', markersize=6, capsize=4)

plt.xlabel('Polynomial Degree')
plt.ylabel('Coefficient Value')
plt.title('Coefficient Evolution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
#%%

#best_degree=6


plt.errorbar(final_bin_centers, final_binned_velocities, 
             yerr=final_binned_velocity_errors,
             fmt='ko', capsize=5, markersize=2, capthick=0.5, label='Binned data')
plt.scatter(radii, velocities, alpha=0.2, s=1, color='gray', label='Individual stars')

if best_degree is not None:
    best_result = fit_results[best_degree]
    v_fine = power_series_fit(r_fine, *best_result['coefficients'])
    v_fit = power_series_fit(final_bin_centers, *best_result['coefficients'])
    
    plt.plot(r_fine, v_fine, 'red', linewidth=1.5, 
            label=f'Best fit (degree {best_degree})')
    
plt.xlabel('Radial Distance (kpc)')
plt.ylabel('Rotational Velocity (km/s)')
plt.title('Best Power Series Fit')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


plt.scatter(radii, velocities, alpha=0.2, s=1, color='gray', label='Individual stars')

if best_degree is not None:
    best_result = fit_results[best_degree]
    v_fine = power_series_fit(r_fine, *best_result['coefficients'])
    v_fit = power_series_fit(final_bin_centers, *best_result['coefficients'])
    
    plt.plot(r_fine, v_fine, 'red', linewidth=1.5, 
            label=f'Best fit (degree {best_degree})')
    
plt.xlabel('Radial Distance (kpc)')
plt.ylabel('Rotational Velocity (km/s)')
plt.title('Best Power Series Fit')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#%%

#Both 4 and 5 are good
best_degree=4
best_degree=5 #5 fits better physically
# Save the best fit results
if best_degree is not None:
    best_result = fit_results[best_degree]
    
    # Create detailed fit results
    fit_data = pd.DataFrame({
        'radius_kpc': final_bin_centers,
        'observed_velocity_kms': final_binned_velocities,
        'velocity_error_kms': final_binned_velocity_errors,
        'fitted_velocity_kms': power_series_fit(final_bin_centers, *best_result['coefficients']),
        'residuals_kms': final_binned_velocities - power_series_fit(final_bin_centers, *best_result['coefficients'])
    })
    
    #fit_data.to_csv("power_series_fit_results_30bins.csv", index=False)
    
    # Save fit parameters
    fit_params = pd.DataFrame({
        'coefficient': [f'a{i}' for i in range(len(best_result['coefficients']))],
        'value': best_result['coefficients'],
        'error': best_result['parameter_errors']
    })
    
    fit_params.to_csv("power_series_coefficients_4_nojeans30bins.csv", index=False)
    
    print(f"\nSaved fitting results:")
    print(f"- power_series_fit_results.csv (data and fits)")
    print(f"- power_series_coefficients.csv (coefficients)")
    
    # Print the best fit equation
    print(f"\nBest fit equation (degree {best_degree}):")
    equation = "v(r) = "
    for i, coeff in enumerate(best_result['coefficients']):
        if i == 0:
            equation += f"{coeff:.4f}"
        elif i == 1:
            equation += f" + {coeff:.4f}*r" if coeff >= 0 else f" - {abs(coeff):.4f}*r"
        else:
            if coeff >= 0:
                equation += f" + {coeff:.4f}*r^{i}"
            else:
                equation += f" - {abs(coeff):.4f}*r^{i}"
    print(equation)

print("\nPower series fitting complete!")


#%%

"""
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


"""
#%%

# ...existing code...

def power_series_fit_no_intercept(r, *coeffs):
    """
    Power series function with NO constant term: v(r) = a1*r + a2*r^2 + a3*r^3 + ...
    Forces y-intercept to be 0
    
    Parameters:
    -----------
    r : array_like
        Radial distances
    *coeffs : tuple
        Coefficients of the power series (a1, a2, a3, ...) - NO a0 term
    
    Returns:
    --------
    v : array_like
        Velocity values at radii r
    """
    result = np.zeros_like(r)
    for i, coeff in enumerate(coeffs):
        result += coeff * r**(i+1)  # Start from r^1, not r^0
    return result

def fit_power_series_zero_intercept(radii, velocities, velocity_errors, max_degree=4):
    """
    Fit a power series to rotation curve data with y-intercept locked at 0
    
    Parameters:
    -----------
    radii : array_like
        Radial distances of bins
    velocities : array_like
        Velocity values of bins
    velocity_errors : array_like
        Uncertainty in velocities
    max_degree : int
        Maximum degree of polynomial to fit
        
    Returns:
    --------
    dict with fitting results for different degrees
    """
    
    results = {}
    
    for degree in range(1, max_degree + 1):
        try:
            # Define the function for this degree (no constant term)
            def poly_func(r, *coeffs):
                return power_series_fit_no_intercept(r, *coeffs)
            
            # Initial guess - now one fewer coefficient (no a0)
            if degree == 1:
                initial_guess = [10]  # Just a1 (slope)
            elif degree == 2:
                initial_guess = [10, -0.1]  # a1, a2
            else:
                initial_guess = [10] + [0.0] * (degree - 1)  # a1, a2, a3, ...
            
            # Fit the curve with error weighting
            popt, pcov = curve_fit(
                poly_func, radii, velocities, 
                sigma=velocity_errors, 
                p0=initial_guess,
                absolute_sigma=True,
                maxfev=5000
            )
            
            # Calculate R-squared and reduced chi-squared
            y_pred = power_series_fit_no_intercept(radii, *popt)
            ss_res = np.sum((velocities - y_pred)**2)
            ss_tot = np.sum((velocities - np.mean(velocities))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Reduced chi-squared
            chi_squared = np.sum(((velocities - y_pred) / velocity_errors)**2)
            reduced_chi_squared = chi_squared / (len(velocities) - len(popt))
            
            # Calculate parameter uncertainties
            param_errors = np.sqrt(np.diag(pcov))
            
            # Store results
            results[degree] = {
                'coefficients': popt,
                'covariance': pcov,
                'parameter_errors': param_errors,
                'r_squared': r_squared,
                'reduced_chi_squared': reduced_chi_squared,
                'aic': len(velocities) * np.log(ss_res/len(velocities)) + 2*len(popt),
                'bic': len(velocities) * np.log(ss_res/len(velocities)) + len(popt)*np.log(len(velocities))
            }
            
        except Exception as e:
            print(f"Failed to fit degree {degree} polynomial: {e}")
            results[degree] = None
    
    return results

# Perform power series fitting with zero intercept
print("Fitting power series to binned rotation curve (y-intercept = 0)...")

fit_results_zero = fit_power_series_zero_intercept(final_bin_centers, final_binned_velocities, 
                                                  final_binned_velocity_errors, max_degree=5)

# Print fitting results
print(f"\n=== Power Series Fitting Results (Zero Intercept) ===")
for degree, result in fit_results_zero.items():
    if result is not None:
        print(f"\nDegree {degree} polynomial:")
        print(f"  R²: {result['r_squared']:.4f}")
        print(f"  Reduced χ²: {result['reduced_chi_squared']:.4f}")
        print(f"  AIC: {result['aic']:.2f}")
        print(f"  BIC: {result['bic']:.2f}")
        print("  Coefficients:")
        for i, (coeff, error) in enumerate(zip(result['coefficients'], result['parameter_errors'])):
            print(f"    a{i+1}: {coeff:.6f} ± {error:.6f}")  # Start from a1, not a0

# Select best fit
best_degree_zero = select_best_fit(fit_results_zero)
print(f"\nBest fit (zero intercept): Degree {best_degree_zero} polynomial")

#%%
# Plot comparison between regular fit and zero-intercept fit
plt.figure(figsize=(15, 10))

#best_degree_zero=5

# Plot 1: Original fit vs Zero-intercept fit
plt.subplot(2, 2, 1)
plt.errorbar(final_bin_centers, final_binned_velocities, 
             yerr=final_binned_velocity_errors,
             fmt='ko', capsize=5, markersize=6, label='Binned data')

r_fine = np.linspace(0, final_bin_centers.max(), 300)  # Start from 0 to show intercept

# Original fit (with intercept)
if best_degree is not None:
    best_result_orig = fit_results[best_degree]
    v_fine_orig = power_series_fit(r_fine, *best_result_orig['coefficients'])
    plt.plot(r_fine, v_fine_orig, 'blue', linewidth=2, 
             label=f'With intercept (deg {best_degree})')

# Zero-intercept fit
if best_degree_zero is not None:
    best_result_zero = fit_results_zero[best_degree_zero]
    v_fine_zero = power_series_fit_no_intercept(r_fine, *best_result_zero['coefficients'])
    plt.plot(r_fine, v_fine_zero, 'red', linewidth=2, 
             label=f'Zero intercept (deg {best_degree_zero})')

plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
plt.xlabel('Radial Distance (kpc)')
plt.ylabel('Rotational Velocity (km/s)')
plt.title('Comparison: With vs Without Intercept')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Residuals comparison
plt.subplot(2, 2, 2)
if best_degree_zero is not None:
    residuals_zero = final_binned_velocities - power_series_fit_no_intercept(final_bin_centers, *best_result_zero['coefficients'])
    plt.errorbar(final_bin_centers, residuals_zero, 
                yerr=final_binned_velocity_errors,
                fmt='ro', capsize=5, markersize=6, label='Zero intercept')

if best_degree is not None:
    residuals_orig = final_binned_velocities - power_series_fit(final_bin_centers, *best_result_orig['coefficients'])
    plt.errorbar(final_bin_centers + 0.05, residuals_orig,  # Offset slightly for visibility
                yerr=final_binned_velocity_errors,
                fmt='bo', capsize=5, markersize=6, label='With intercept')

plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
plt.xlabel('Radial Distance (kpc)')
plt.ylabel('Residuals (km/s)')
plt.title('Residuals Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Zero-intercept fit alone with data
plt.subplot(2, 2, 3)
plt.scatter(radii, velocities, alpha=0.1, s=1, color='gray', label='Individual stars')
plt.errorbar(final_bin_centers, final_binned_velocities, 
             yerr=final_binned_velocity_errors,
             fmt='ko', capsize=5, markersize=8, label='Binned data')

if best_degree_zero is not None:
    plt.plot(r_fine, v_fine_zero, 'red', linewidth=3, 
             label=f'Zero intercept fit (degree {best_degree_zero})')

plt.xlabel('Radial Distance (kpc)')
plt.ylabel('Rotational Velocity (km/s)')
plt.title('Zero-Intercept Power Series Fit')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Model comparison metrics
plt.subplot(2, 2, 4)
degrees_orig = []
degrees_zero = []
aics_orig = []
aics_zero = []

for degree, result in fit_results.items():
    if result is not None:
        degrees_orig.append(degree)
        aics_orig.append(result['aic'])

for degree, result in fit_results_zero.items():
    if result is not None:
        degrees_zero.append(degree)
        aics_zero.append(result['aic'])

plt.plot(degrees_orig, aics_orig, 'bo-', label='With intercept', markersize=8)
plt.plot(degrees_zero, aics_zero, 'ro-', label='Zero intercept', markersize=8)
plt.xlabel('Polynomial Degree')
plt.ylabel('AIC (lower is better)')
plt.title('Model Selection: AIC Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%%
# Save zero-intercept fit results
best_degree_zero=5
if best_degree_zero is not None:
    best_result_zero = fit_results_zero[best_degree_zero]
    
    # Create detailed fit results
    fit_data_zero = pd.DataFrame({
        'radius_kpc': final_bin_centers,
        'observed_velocity_kms': final_binned_velocities,
        'velocity_error_kms': final_binned_velocity_errors,
        'fitted_velocity_kms': power_series_fit_no_intercept(final_bin_centers, *best_result_zero['coefficients']),
        'residuals_kms': final_binned_velocities - power_series_fit_no_intercept(final_bin_centers, *best_result_zero['coefficients'])
    })
    
    fit_data_zero.to_csv("power_series_fit_results_zero_intercept_5_bin1.csv", index=False)
    
    # Save fit parameters
    fit_params_zero = pd.DataFrame({
        'coefficient': [f'a{i+1}' for i in range(len(best_result_zero['coefficients']))],  # a1, a2, a3, ...
        'value': best_result_zero['coefficients'],
        'error': best_result_zero['parameter_errors']
    })
    
    fit_params_zero.to_csv("power_series_coefficients_zero_intercept_5_bin1.csv", index=False)
    
    print(f"\nSaved zero-intercept fitting results:")
    print(f"- power_series_fit_results_zero_intercept.csv")
    print(f"- power_series_coefficients_zero_intercept.csv")
    
    # Print the zero-intercept fit equation
    print(f"\nZero-intercept fit equation (degree {best_degree_zero}):")
    equation = "v(r) = "
    for i, coeff in enumerate(best_result_zero['coefficients']):
        power = i + 1  # Start from r^1
        if i == 0:
            equation += f"{coeff:.6f}*r"
        else:
            if coeff >= 0:
                equation += f" + {coeff:.6f}*r^{power}"
            else:
                equation += f" - {abs(coeff):.6f}*r^{power}"
    print(equation)
    
    print(f"\nFit statistics (zero intercept):")
    print(f"R² = {best_result_zero['r_squared']:.6f}")
    print(f"Reduced χ² = {best_result_zero['reduced_chi_squared']:.3f}")

print("\nZero-intercept power series fitting complete!")
