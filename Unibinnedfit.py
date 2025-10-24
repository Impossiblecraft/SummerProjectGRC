import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the original data
data = pd.read_csv("rotcurvebin80k1.csv")

radii = data['pos_kpc'].values
velocities = data['rotv_kms'].values
radii_err = data['pos_err_kpc'].values
velocities_err = data['rotv_err_kms'].values

print(f"Loaded {len(radii)} stars for unbinned fitting")

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

def fit_unbinned_power_series(radii, velocities, velocity_errors, max_degree=6):
    """
    Fit a power series to unbinned rotation curve data
    
    Parameters:
    -----------
    radii : array_like
        Radial distances of stars
    velocities : array_like
        Velocity values of stars
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
            
            # Initial guess (start with reasonable values)
            if degree == 1:
                initial_guess = [200, 10]  # v0 + slope
            elif degree == 2:
                initial_guess = [200, 10, -0.1]  # add curvature
            else:
                initial_guess = [200, 10] + [0.0] * (degree - 1)
            
            # Fit the curve with error weighting
            popt, pcov = curve_fit(
                poly_func, radii, velocities, 
                sigma=velocity_errors, 
                p0=initial_guess,
                absolute_sigma=True,
                maxfev=10000
            )
            
            # Calculate goodness of fit metrics
            y_pred = power_series_fit(radii, *popt)
            residuals = velocities - y_pred
            
            # R-squared
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((velocities - np.mean(velocities))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Reduced chi-squared
            chi_squared = np.sum((residuals / velocity_errors)**2)
            reduced_chi_squared = chi_squared / (len(velocities) - len(popt))
            
            # Calculate parameter uncertainties
            param_errors = np.sqrt(np.diag(pcov))
            
            # Information criteria
            n = len(velocities)
            k = len(popt)
            aic = n * np.log(ss_res/n) + 2*k
            bic = n * np.log(ss_res/n) + k*np.log(n)
            
            # Store results
            results[degree] = {
                'coefficients': popt,
                'covariance': pcov,
                'parameter_errors': param_errors,
                'r_squared': r_squared,
                'reduced_chi_squared': reduced_chi_squared,
                'aic': aic,
                'bic': bic,
                'residuals': residuals,
                'fitted_values': y_pred
            }
            
            print(f"Degree {degree}: R² = {r_squared:.4f}, χ²_red = {reduced_chi_squared:.2f}")
            
        except Exception as e:
            print(f"Failed to fit degree {degree} polynomial: {e}")
            results[degree] = None
    
    return results

# Perform unbinned fitting
print("Fitting power series to unbinned data...")
unbinned_results = fit_unbinned_power_series(radii, velocities, velocities_err, max_degree=5)

# Select best fit
def select_best_unbinned_fit(fit_results):
    """Select best fit based on BIC (good for large datasets)"""
    valid_results = {k: v for k, v in fit_results.items() if v is not None}
    
    if not valid_results:
        return None
    
    best_degree = min(valid_results.keys(), 
                     key=lambda k: valid_results[k]['bic'])
    return best_degree

best_unbinned_degree = select_best_unbinned_fit(unbinned_results)
print(f"\nBest unbinned fit: Degree {best_unbinned_degree} polynomial")

if best_unbinned_degree is not None:
    best_result = unbinned_results[best_unbinned_degree]
    print(f"Best fit statistics:")
    print(f"  R² = {best_result['r_squared']:.6f}")
    print(f"  Reduced χ² = {best_result['reduced_chi_squared']:.3f}")
    print(f"  BIC = {best_result['bic']:.1f}")

#%%
# Create comprehensive plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Scatter plot with best fit
ax1 = axes[0, 0]
ax1.scatter(radii, velocities, alpha=0.3, s=2, color='gray', label=f'Individual stars (n={len(radii)})')


if best_unbinned_degree is not None:
    best_result = unbinned_results[best_unbinned_degree]
    
    # Create smooth curve for plotting
    r_smooth = np.linspace(radii.min(), radii.max(), 500)
    v_smooth = power_series_fit(r_smooth, *best_result['coefficients'])
    
    ax1.plot(r_smooth, v_smooth, 'red', linewidth=3, 
             label=f'Degree {best_unbinned_degree} fit (R²={best_result["r_squared"]:.4f})')

ax1.set_xlabel('Radial Distance (kpc)')
ax1.set_ylabel('Rotational Velocity (km/s)')
ax1.set_title('Unbinned Data with Power Series Fit')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals vs radius
ax2 = axes[0, 1]
if best_unbinned_degree is not None:
    residuals = best_result['residuals']
    ax2.scatter(radii, residuals, alpha=0.3, s=2, color='blue')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    # Add running mean of residuals to show systematic trends
    from scipy.ndimage import uniform_filter1d
    sorted_indices = np.argsort(radii)
    sorted_radii = radii[sorted_indices]
    sorted_residuals = residuals[sorted_indices]
    
    # Running mean with window size proportional to data density
    window_size = max(50, len(radii) // 100)
    if len(sorted_residuals) > window_size:
        running_mean = uniform_filter1d(sorted_residuals, size=window_size, mode='nearest')
        ax2.plot(sorted_radii, running_mean, 'orange', linewidth=2, label='Running mean')
        ax2.legend()

ax2.set_xlabel('Radial Distance (kpc)')
ax2.set_ylabel('Residuals (km/s)')
ax2.set_title('Fit Residuals')
ax2.grid(True, alpha=0.3)

# Plot 3: Model comparison
ax3 = axes[1, 0]
degrees = []
bics = []
r_squareds = []

for degree, result in unbinned_results.items():
    if result is not None:
        degrees.append(degree)
        bics.append(result['bic'])
        r_squareds.append(result['r_squared'])

ax3_twin = ax3.twinx()
line1 = ax3.plot(degrees, bics, 'bo-', label='BIC', markersize=8)
line2 = ax3_twin.plot(degrees, r_squareds, 'ro-', label='R²', markersize=8)

ax3.set_xlabel('Polynomial Degree')
ax3.set_ylabel('BIC (lower is better)', color='blue')
ax3_twin.set_ylabel('R² (higher is better)', color='red')
ax3.set_title('Model Selection')
ax3.grid(True, alpha=0.3)

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, loc='center right')

# Plot 4: Histogram of residuals
ax4 = axes[1, 1]
if best_unbinned_degree is not None:
    residuals = best_result['residuals']
    ax4.hist(residuals, bins=50, alpha=0.7, color='green', density=True, label='Residuals')
    
    # Overlay normal distribution for comparison
    mu, sigma = np.mean(residuals), np.std(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    normal_dist = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
    ax4.plot(x, normal_dist, 'red', linewidth=2, label=f'Normal (μ={mu:.1f}, σ={sigma:.1f})')
    
    ax4.axvline(x=0, color='black', linestyle='--', alpha=0.7)

ax4.set_xlabel('Residuals (km/s)')
ax4.set_ylabel('Density')
ax4.set_title('Residual Distribution')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%%
# Print detailed results and equation
if best_unbinned_degree is not None:
    best_result = unbinned_results[best_unbinned_degree]
    
    print(f"\n=== Unbinned Power Series Fit Results ===")
    print(f"Best fit: Degree {best_unbinned_degree} polynomial")
    print(f"Number of data points: {len(radii)}")
    print(f"R² = {best_result['r_squared']:.6f}")
    print(f"Reduced χ² = {best_result['reduced_chi_squared']:.3f}")
    print(f"RMS residual = {np.std(best_result['residuals']):.2f} km/s")
    
    print(f"\nCoefficients:")
    for i, (coeff, error) in enumerate(zip(best_result['coefficients'], best_result['parameter_errors'])):
        print(f"  a{i} = {coeff:.6f} ± {error:.6f}")
    
    # Print equation
    print(f"\nFit equation:")
    equation = "v(r) = "
    for i, coeff in enumerate(best_result['coefficients']):
        if i == 0:
            equation += f"{coeff:.4f}"
        elif i == 1:
            if coeff >= 0:
                equation += f" + {coeff:.4f}*r"
            else:
                equation += f" - {abs(coeff):.4f}*r"
        else:
            if coeff >= 0:
                equation += f" + {coeff:.6f}*r^{i}"
            else:
                equation += f" - {abs(coeff):.6f}*r^{i}"
    print(equation)
    
    # Save results
    unbinned_fit_data = pd.DataFrame({
        'radius_kpc': radii,
        'observed_velocity_kms': velocities,
        'velocity_error_kms': velocities_err,
        'fitted_velocity_kms': best_result['fitted_values'],
        'residuals_kms': best_result['residuals']
    })
    
    unbinned_fit_data.to_csv("unbinned_power_series_fit.csv", index=False)
    
    # Save coefficients
    unbinned_coeffs = pd.DataFrame({
        'coefficient': [f'a{i}' for i in range(len(best_result['coefficients']))],
        'value': best_result['coefficients'],
        'error': best_result['parameter_errors']
    })
    
    unbinned_coeffs.to_csv("unbinned_power_series_coefficients.csv", index=False)
    
    print(f"\nSaved results:")
    print(f"- unbinned_power_series_fit.csv")
    print(f"- unbinned_power_series_coefficients.csv")

print("\nUnbinned power series fitting complete!")
#%%
# ...existing code...

def calculate_isolation_weights(radii, scale_factor=1.0):
    """
    Calculate weights based on isolation in x-direction (radius)
    Points that are more isolated get higher weights
    
    Parameters:
    -----------
    radii : array_like
        Radial distances of stars
    scale_factor : float
        Controls how much to emphasize isolation (higher = more emphasis)
        
    Returns:
    --------
    weights : array_like
        Isolation weights (higher for isolated points)
    """
    n = len(radii)
    weights = np.ones(n)
    
    # Sort radii to find nearest neighbors efficiently
    sorted_indices = np.argsort(radii)
    sorted_radii = radii[sorted_indices]
    
    for i, idx in enumerate(sorted_indices):
        # Find distance to nearest neighbors
        distances = []
        
        # Distance to left neighbor
        if i > 0:
            distances.append(sorted_radii[i] - sorted_radii[i-1])
        
        # Distance to right neighbor  
        if i < n-1:
            distances.append(sorted_radii[i+1] - sorted_radii[i])
            
        # If we have neighbors, use minimum distance as isolation measure
        if distances:
            min_neighbor_distance = min(distances)
            # Weight is proportional to isolation (larger gap = higher weight)
            weights[idx] = 1.0 + scale_factor * min_neighbor_distance
        else:
            # Single point - give maximum weight
            weights[idx] = 1.0 + scale_factor * 1.0
    
    return weights

def fit_unbinned_power_series_isolated(radii, velocities, velocity_errors, max_degree=6, isolation_scale=1.0):
    """
    Fit a power series to unbinned rotation curve data
    with higher weights for x-isolated points
    
    Parameters:
    -----------
    radii : array_like
        Radial distances of stars
    velocities : array_like
        Velocity values of stars
    velocity_errors : array_like
        Uncertainty in velocities
    max_degree : int
        Maximum degree of polynomial to fit
    isolation_scale : float
        How much to emphasize isolated points
        
    Returns:
    --------
    dict with fitting results for different degrees
    """
    
    # Calculate isolation weights
    isolation_weights = calculate_isolation_weights(radii, isolation_scale)
    
    # Combine with inverse error weights
    # Total weight = isolation_weight / velocity_error^2
    total_weights = isolation_weights / (velocity_errors**2)
    
    results = {}
    
    for degree in range(1, max_degree + 1):
        try:
            # Define the function for this degree
            def poly_func(r, *coeffs):
                return power_series_fit(r, *coeffs)
            
            # Initial guess (start with reasonable values)
            if degree == 1:
                initial_guess = [200, 10]  # v0 + slope
            elif degree == 2:
                initial_guess = [200, 10, -0.1]  # add curvature
            else:
                initial_guess = [200, 10] + [0.0] * (degree - 1)
            
            # Fit the curve with isolation weighting
            # Use sqrt of weights as sigma (curve_fit uses 1/sigma^2 as weights)
            effective_sigma = 1.0 / np.sqrt(total_weights)
            
            popt, pcov = curve_fit(
                poly_func, radii, velocities, 
                sigma=effective_sigma,
                p0=initial_guess,
                absolute_sigma=True,
                maxfev=10000
            )
            
            # Calculate goodness of fit metrics
            y_pred = power_series_fit(radii, *popt)
            residuals = velocities - y_pred
            
            # Weighted residuals for isolated points
            weighted_residuals = residuals * np.sqrt(total_weights)
            
            # R-squared (unweighted for interpretability)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((velocities - np.mean(velocities))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Weighted R-squared
            weighted_ss_res = np.sum(weighted_residuals**2)
            weighted_mean_v = np.sum(velocities * total_weights) / np.sum(total_weights)
            weighted_ss_tot = np.sum(total_weights * (velocities - weighted_mean_v)**2)
            weighted_r_squared = 1 - (weighted_ss_res / weighted_ss_tot)
            
            # Reduced chi-squared with isolation weighting
            chi_squared = np.sum(weighted_residuals**2)
            reduced_chi_squared = chi_squared / (len(velocities) - len(popt))
            
            # Calculate parameter uncertainties
            param_errors = np.sqrt(np.diag(pcov))
            
            # Information criteria (using weighted residuals)
            n = len(velocities)
            k = len(popt)
            aic = n * np.log(weighted_ss_res/n) + 2*k
            bic = n * np.log(weighted_ss_res/n) + k*np.log(n)
            
            # Store results
            results[degree] = {
                'coefficients': popt,
                'covariance': pcov,
                'parameter_errors': param_errors,
                'r_squared': r_squared,
                'weighted_r_squared': weighted_r_squared,
                'reduced_chi_squared': reduced_chi_squared,
                'aic': aic,
                'bic': bic,
                'residuals': residuals,
                'weighted_residuals': weighted_residuals,
                'fitted_values': y_pred,
                'isolation_weights': isolation_weights,
                'total_weights': total_weights
            }
            
            print(f"Degree {degree}: R² = {r_squared:.4f}, Weighted R² = {weighted_r_squared:.4f}")
            
        except Exception as e:
            print(f"Failed to fit degree {degree} polynomial: {e}")
            results[degree] = None
    
    return results

# Perform unbinned fitting with isolation weighting
print("Fitting power series to unbinned data (emphasizing isolated points)...")
unbinned_results = fit_unbinned_power_series_isolated(radii, velocities, velocities_err, 
                                                     max_degree=10, isolation_scale=2.0)

# Select best fit based on weighted BIC
def select_best_isolated_fit(fit_results):
    """Select best fit based on BIC with isolation weighting"""
    valid_results = {k: v for k, v in fit_results.items() if v is not None}
    
    if not valid_results:
        return None
    
    best_degree = min(valid_results.keys(), 
                     key=lambda k: valid_results[k]['bic'])
    return best_degree

best_unbinned_degree = select_best_isolated_fit(unbinned_results)
print(f"\nBest isolated-weighted fit: Degree {best_unbinned_degree} polynomial")

if best_unbinned_degree is not None:
    best_result = unbinned_results[best_unbinned_degree]
    print(f"Best fit statistics:")
    print(f"  R² = {best_result['r_squared']:.6f}")
    print(f"  Weighted R² = {best_result['weighted_r_squared']:.6f}")
    print(f"  Reduced χ² = {best_result['reduced_chi_squared']:.3f}")
    print(f"  BIC = {best_result['bic']:.1f}")

#%%
# Visualize isolation weights
if best_unbinned_degree is not None:
    best_result = unbinned_results[best_unbinned_degree]
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Data with isolation weights shown as point sizes
    plt.subplot(2, 3, 1)
    isolation_weights = best_result['isolation_weights']
    # Normalize weights for plotting (size between 1 and 50)
    norm_weights = 1 + 49 * (isolation_weights - isolation_weights.min()) / (isolation_weights.max() - isolation_weights.min())
    
    scatter = plt.scatter(radii, velocities, s=norm_weights, alpha=0.6, c=isolation_weights, 
                         cmap='viridis', label='Data (size ∝ isolation weight)')
    plt.colorbar(scatter, label='Isolation Weight')
    
    # Plot fit
    r_smooth = np.linspace(radii.min(), radii.max(), 500)
    v_smooth = power_series_fit(r_smooth, *best_result['coefficients'])
    plt.plot(r_smooth, v_smooth, 'red', linewidth=3, label=f'Degree {best_unbinned_degree} fit')
    
    plt.xlabel('Radial Distance (kpc)')
    plt.ylabel('Rotational Velocity (km/s)')
    plt.title('Data with Isolation Weighting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Isolation weights vs radius
    plt.subplot(2, 3, 2)
    plt.scatter(radii, isolation_weights, alpha=0.6, s=10)
    plt.xlabel('Radial Distance (kpc)')
    plt.ylabel('Isolation Weight')
    plt.title('Isolation Weight vs Radius')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Weighted residuals
    plt.subplot(2, 3, 3)
    plt.scatter(radii, best_result['weighted_residuals'], alpha=0.6, s=10)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Radial Distance (kpc)')
    plt.ylabel('Weighted Residuals')
    plt.title('Weighted Residuals')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Regular residuals
    plt.subplot(2, 3, 4)
    plt.scatter(radii, best_result['residuals'], alpha=0.6, s=10)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Radial Distance (kpc)')
    plt.ylabel('Residuals (km/s)')
    plt.title('Unweighted Residuals')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Comparison of fits
    plt.subplot(2, 3, 5)
    plt.scatter(radii, velocities, alpha=0.3, s=2, color='gray', label='Data')
    plt.plot(r_smooth, v_smooth, 'red', linewidth=3, 
             label=f'Isolation-weighted (R²={best_result["weighted_r_squared"]:.4f})')
    plt.xlabel('Radial Distance (kpc)')
    plt.ylabel('Rotational Velocity (km/s)')
    plt.title('Isolation-Weighted Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Weight distribution histogram
    plt.subplot(2, 3, 6)
    plt.hist(isolation_weights, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Isolation Weight')
    plt.ylabel('Number of Points')
    plt.title('Distribution of Isolation Weights')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

print(f"\nIsolation weighting gives higher influence to points in sparse regions of radius")
print(f"This helps ensure the fit represents the full radial range more equally")