import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Wrong, need to fix this whole code

# Load the raw data with Cartesian coordinates
data = pd.read_csv("ProcessedVicinityWithUncertainties.csv")

print(f"Loaded {len(data)} stars from rawrotcurve.csv")
print("Columns available:", data.columns.tolist())

# Extract Cartesian coordinates and velocities with their errors
x = data['px'].values
y = data['py'].values
z = data['py'].values
vx = data['vx'].values
vy = data['vy'].values
vz = data['vz'].values

# Extract uncertainties
x_err = data['px_err'].values
y_err = data['py_err'].values
z_err = data['pz_err'].values
vx_err = data['vx_err'].values
vy_err = data['vy_err'].values
vz_err = data['vz_err'].values

# ...existing code...

def theoretical_stellar_density(R, z=0, model='exponential_disk'):
    """
    Calculate theoretical stellar number density using standard galactic models
    
    Parameters:
    -----------
    R : array_like
        Radial distance from galactic center (kpc)
    z : array_like or float
        Height above galactic plane (kpc)
    model : str
        Density model to use: 'exponential_disk', 'sersic', 'double_exponential'
    
    Returns:
    --------
    density : array_like
        Stellar number density (arbitrary units)
    """
    
    if model == 'exponential_disk':
        # Simple exponential disk: ρ(R,z) = ρ₀ * exp(-R/R_d) * exp(-|z|/z_d)
        R_d = 2.6  # kpc - disk scale length (typical for Milky Way)
        z_d = 0.3  # kpc - disk scale height
        rho_0 = 1.0  # Normalization constant
        
        density = rho_0 * np.exp(-R / R_d) * np.exp(-np.abs(z) / z_d)
        
    elif model == 'sersic':
        # Sersic profile: ρ(R) = ρ₀ * exp(-b_n * (R/R_e)^(1/n))
        R_e = 2.5  # kpc - effective radius
        n = 1.0    # Sersic index (n=1 is exponential)
        b_n = 1.678  # For n=1
        rho_0 = 1.0
        
        density = rho_0 * np.exp(-b_n * (R / R_e)**(1/n))
        
    elif model == 'double_exponential':
        # Double exponential (thin + thick disk)
        # ρ(R,z) = ρ_thin * exp(-R/R_thin) * exp(-|z|/z_thin) + ρ_thick * exp(-R/R_thick) * exp(-|z|/z_thick)
        
        # Thin disk parameters
        R_thin = 2.6  # kpc
        z_thin = 0.3  # kpc
        rho_thin = 0.85  # Relative normalization
        
        # Thick disk parameters
        R_thick = 3.6  # kpc
        z_thick = 0.9  # kpc
        rho_thick = 0.15  # Relative normalization
        
        density_thin = rho_thin * np.exp(-R / R_thin) * np.exp(-np.abs(z) / z_thin)
        density_thick = rho_thick * np.exp(-R / R_thick) * np.exp(-np.abs(z) / z_thick)
        
        density = density_thin + density_thick
        
    elif model == 'miyamoto_nagai':
        # Miyamoto-Nagai disk model
        a = 3.0  # kpc - scale parameter
        b = 0.28 # kpc - scale height parameter
        
        # Convert to cylindrical coordinates if needed
        z = np.atleast_1d(z)
        R = np.atleast_1d(R)
        
        # Miyamoto-Nagai density
        sqrt_term = np.sqrt(z**2 + b**2)
        denominator = (R**2 + (a + sqrt_term)**2)**(5/2) * (z**2 + b**2)**(3/2)
        
        # Simplified form for stellar density (not mass density)
        density = 1.0 / (1 + (R/a)**2 + (np.abs(z)/b)**2)**2
        
    else:
        raise ValueError(f"Unknown density model: {model}")
    
    return density

def true_jeans_equation_with_model(x, y, z, vx, vy, vz, 
                                  x_err, y_err, z_err, vx_err, vy_err, vz_err,
                                  density_model='double_exponential'):
    """
    Calculate circular velocity using TRUE Jeans equation with theoretical stellar density
    
    The Jeans equation in cylindrical coordinates:
    1/ρ * d(ρ σ_R²)/dR + (σ_R² - σ_φ²)/R = -dΦ/dR = v_c²/R
    
    Therefore: v_c² = σ_φ² + (σ_R² - σ_φ²) + R * [1/ρ * d(ρ σ_R²)/dR]
    """
    
    # Step 1: Convert to cylindrical coordinates
    R = np.sqrt(x**2 + y**2)
    R_err = np.sqrt((x * x_err)**2 + (y * y_err)**2) / R
    
    # Unit vectors
    e_R_x, e_R_y = x / R, y / R
    e_phi_x, e_phi_y = -y / R, x / R
    
    # Velocity components
    v_radial = vx * e_R_x + vy * e_R_y
    v_tangential = vx * e_phi_x + vy * e_phi_y
    
    # Step 2: Create radial grid for analysis
    R_min, R_max = np.percentile(R, [5, 95])
    R_grid = np.linspace(R_min, R_max, 100)
    
    # Step 3: Calculate theoretical stellar density on grid
    # Use mean z-height for each radial bin
    z_grid = np.zeros_like(R_grid)
    for i, r in enumerate(R_grid):
        mask = (np.abs(R - r) < 0.5)  # Stars within 0.5 kpc of this radius
        if np.sum(mask) > 0:
            z_grid[i] = np.mean(np.abs(z[mask]))
        else:
            z_grid[i] = 0.3  # Default scale height
    
    stellar_density_grid = theoretical_stellar_density(R_grid, z_grid, model=density_model)
    
    print(f"Using {density_model} stellar density model")
    print(f"Density range: {stellar_density_grid.min():.3e} to {stellar_density_grid.max():.3e}")
    
    # Step 4: Calculate velocity dispersions using sliding window
    def calculate_local_dispersions(R_eval, window_factor=0.2):
        """Calculate local velocity dispersions at radius R_eval"""
        # Adaptive window size (larger for sparse regions)
        base_window = (R_max - R_min) / 20
        window_size = base_window * (1 + window_factor * R_eval / R_max)
        
        # Find stars within the window
        mask = np.abs(R - R_eval) <= window_size
        
        if np.sum(mask) < 15:  # Need minimum number of stars
            return np.nan, np.nan, 0
        
        # Calculate dispersions for stars in this window
        v_r_local = v_radial[mask]
        v_phi_local = v_tangential[mask]
        
        # Remove outliers (3-sigma clipping)
        v_r_median = np.median(v_r_local)
        v_r_std = np.std(v_r_local)
        good_r = np.abs(v_r_local - v_r_median) < 3 * v_r_std
        
        v_phi_median = np.median(v_phi_local)
        v_phi_std = np.std(v_phi_local)
        good_phi = np.abs(v_phi_local - v_phi_median) < 3 * v_phi_std
        
        # Use cleaned data
        if np.sum(good_r) > 5 and np.sum(good_phi) > 5:
            sigma_R_sq = np.var(v_r_local[good_r])
            sigma_phi_sq = np.var(v_phi_local[good_phi])
            n_local = min(np.sum(good_r), np.sum(good_phi))
        else:
            sigma_R_sq = np.var(v_r_local)
            sigma_phi_sq = np.var(v_phi_local)
            n_local = len(v_r_local)
        
        return sigma_R_sq, sigma_phi_sq, n_local
    
    # Calculate dispersions on the radial grid
    sigma_R_squared_grid = np.zeros(len(R_grid))
    sigma_phi_squared_grid = np.zeros(len(R_grid))
    n_stars_grid = np.zeros(len(R_grid))
    
    for i, r in enumerate(R_grid):
        sigma_R_sq, sigma_phi_sq, n_local = calculate_local_dispersions(r)
        sigma_R_squared_grid[i] = sigma_R_sq if not np.isnan(sigma_R_sq) else 0
        sigma_phi_squared_grid[i] = sigma_phi_sq if not np.isnan(sigma_phi_sq) else 0
        n_stars_grid[i] = n_local
    
    # Smooth the dispersion profiles to reduce noise
    from scipy.ndimage import gaussian_filter1d
    valid_mask = n_stars_grid > 0
    
    if np.sum(valid_mask) > 5:
        sigma_R_squared_grid[valid_mask] = gaussian_filter1d(
            sigma_R_squared_grid[valid_mask], sigma=1.0)
        sigma_phi_squared_grid[valid_mask] = gaussian_filter1d(
            sigma_phi_squared_grid[valid_mask], sigma=1.0)
    
    # Step 5: Calculate derivatives for Jeans equation
    # d(ρ σ_R²)/dR
    rho_sigma_R_sq = stellar_density_grid * sigma_R_squared_grid
    
    # Use finite differences for derivative
    d_rho_sigma_R_sq_dR = np.gradient(rho_sigma_R_sq, R_grid)
    
    # Step 6: Apply TRUE Jeans equation
    # v_c² = σ_φ² + (σ_R² - σ_φ²) + R * [1/ρ * d(ρ σ_R²)/dR]
    
    # Avoid division by zero
    rho_nonzero = np.maximum(stellar_density_grid, 1e-10)
    
    # Calculate each term separately for analysis
    term1_sigma_phi_sq = sigma_phi_squared_grid
    term2_asymmetric_drift = sigma_R_squared_grid - sigma_phi_squared_grid
    term3_pressure_gradient = R_grid * d_rho_sigma_R_sq_dR / rho_nonzero
    
    v_circular_squared_jeans = term1_sigma_phi_sq + term2_asymmetric_drift + term3_pressure_gradient
    
    # Ensure positive values
    v_circular_squared_jeans = np.maximum(v_circular_squared_jeans, 0)
    v_circular_jeans_grid = np.sqrt(v_circular_squared_jeans)
    
    # Step 7: Interpolate to individual star positions
    from scipy.interpolate import interp1d
    
    valid_grid_mask = (n_stars_grid > 5) & np.isfinite(v_circular_jeans_grid)
    
    if np.sum(valid_grid_mask) > 3:
        # Create interpolation functions
        interp_vc = interp1d(R_grid[valid_grid_mask], v_circular_jeans_grid[valid_grid_mask], 
                           kind='linear', bounds_error=False, fill_value='extrapolate')
        interp_rho = interp1d(R_grid[valid_grid_mask], stellar_density_grid[valid_grid_mask],
                            kind='linear', bounds_error=False, fill_value='extrapolate')
        interp_sigma_R = interp1d(R_grid[valid_grid_mask], 
                                np.sqrt(sigma_R_squared_grid[valid_grid_mask]),
                                kind='linear', bounds_error=False, fill_value='extrapolate')
        interp_sigma_phi = interp1d(R_grid[valid_grid_mask], 
                                  np.sqrt(sigma_phi_squared_grid[valid_grid_mask]),
                                  kind='linear', bounds_error=False, fill_value='extrapolate')
        
        # Interpolate to star positions
        v_circular_stars = interp_vc(R)
        stellar_density_stars = interp_rho(R)
        sigma_R_stars = interp_sigma_R(R)
        sigma_phi_stars = interp_sigma_phi(R)
        
    else:
        print("Warning: Insufficient data for Jeans equation - using kinematic approximation")
        v_circular_stars = np.abs(v_tangential)
        stellar_density_stars = theoretical_stellar_density(R, np.abs(z), model=density_model)
        sigma_R_stars = np.full(len(R), np.std(v_radial))
        sigma_phi_stars = np.full(len(R), np.std(v_tangential))
    
    # Step 8: Error estimation
    # Tangential velocity error (kinematic part)
    v_tangential_err = np.sqrt(
        ((-y / R) * vx_err)**2 + 
        ((x / R) * vy_err)**2 + 
        ((vy / R - vx * y / R**2) * x_err)**2 + 
        ((-vx / R - vy * x / R**2) * y_err)**2
    )
    
    # Additional uncertainty from Jeans equation
    jeans_systematic_err = 0.15 * v_circular_stars  # 15% systematic uncertainty
    dispersion_err = 0.1 * (sigma_R_stars + sigma_phi_stars)  # Dispersion uncertainty
    
    v_circular_err = np.sqrt(v_tangential_err**2 + jeans_systematic_err**2 + dispersion_err**2)
    
    return {
        'radial_distance': R,
        'radial_distance_err': R_err,
        'circular_velocity_jeans': v_circular_stars,
        'circular_velocity_jeans_err': v_circular_err,
        'tangential_velocity': v_tangential,
        'radial_velocity': v_radial,
        'stellar_density_theoretical': stellar_density_stars,
        'sigma_radial': sigma_R_stars,
        'sigma_tangential': sigma_phi_stars,
        'z_height': z,
        'vertical_velocity': vz,
        # Grid quantities for analysis
        'R_grid': R_grid,
        'stellar_density_grid': stellar_density_grid,
        'v_circular_grid': v_circular_jeans_grid,
        'sigma_R_grid': np.sqrt(sigma_R_squared_grid),
        'sigma_phi_grid': np.sqrt(sigma_phi_squared_grid),
        'jeans_terms': {
            'sigma_phi_squared': term1_sigma_phi_sq,
            'asymmetric_drift': term2_asymmetric_drift,
            'pressure_gradient': term3_pressure_gradient
        },
        'density_model': density_model
    }

# Apply TRUE Jeans equation with theoretical density
print("Calculating circular velocities using TRUE Jeans equation with theoretical density...")

# Test different density models
density_models = ['exponential_disk', 'double_exponential', 'miyamoto_nagai']
jeans_results_models = {}

for model in density_models:
    print(f"\nTesting {model} model...")
    try:
        result = true_jeans_equation_with_model(x, y, z, vx, vy, vz,
                                              x_err, y_err, z_err, 
                                              vx_err, vy_err, vz_err,
                                              density_model=model)
        jeans_results_models[model] = result
        print(f"  Success! v_c range: {result['circular_velocity_jeans'].min():.1f} - {result['circular_velocity_jeans'].max():.1f} km/s")
    except Exception as e:
        print(f"  Failed: {e}")

# Use the best model (double_exponential is most realistic)
best_model = 'double_exponential'
if best_model in jeans_results_models:
    jeans_results = jeans_results_models[best_model]
    print(f"\nUsing {best_model} model for final analysis")
else:
    jeans_results = list(jeans_results_models.values())[0]
    best_model = list(jeans_results_models.keys())[0]
    print(f"\nUsing {best_model} model (fallback)")

#%%
# Plot TRUE Jeans equation with theoretical density
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Theoretical stellar density profile
ax1 = axes[0, 0]
ax1.semilogy(jeans_results['R_grid'], jeans_results['stellar_density_grid'], 'b-', linewidth=3)
ax1.set_xlabel('Radial Distance (kpc)')
ax1.set_ylabel('Stellar Density (theoretical)')
ax1.set_title(f'Theoretical Density: {jeans_results["density_model"]}')
ax1.grid(True, alpha=0.3)

# Plot 2: Velocity dispersions
ax2 = axes[0, 1]
ax2.plot(jeans_results['R_grid'], jeans_results['sigma_R_grid'], 'r-', 
         linewidth=2, label='σ_R (observed)')
ax2.plot(jeans_results['R_grid'], jeans_results['sigma_phi_grid'], 'b-', 
         linewidth=2, label='σ_φ (observed)')
ax2.set_xlabel('Radial Distance (kpc)')
ax2.set_ylabel('Velocity Dispersion (km/s)')
ax2.set_title('Observed Velocity Dispersions')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Jeans equation terms
ax3 = axes[0, 2]
jeans_terms = jeans_results['jeans_terms']
ax3.plot(jeans_results['R_grid'], jeans_terms['sigma_phi_squared'], 'b-', 
         label='σ_φ²', linewidth=2)
ax3.plot(jeans_results['R_grid'], jeans_terms['asymmetric_drift'], 'r-', 
         label='σ_R² - σ_φ²', linewidth=2)
ax3.plot(jeans_results['R_grid'], jeans_terms['pressure_gradient'], 'g-', 
         label='R×d(ρσ_R²)/dR/ρ', linewidth=2)
ax3.set_xlabel('Radial Distance (kpc)')
ax3.set_ylabel('Velocity² Terms (km²/s²)')
ax3.set_title('TRUE Jeans Equation Components')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Rotation curves comparison
ax4 = axes[1, 0]
R_jeans = jeans_results['radial_distance']
v_tang_simple = jeans_results['tangential_velocity']
v_circ_jeans = jeans_results['circular_velocity_jeans']

ax4.scatter(R_jeans, np.abs(v_tang_simple), alpha=0.3, s=2, color='blue', 
           label='Simple |v_φ|')
ax4.scatter(R_jeans, v_circ_jeans, alpha=0.3, s=2, color='red', 
           label='TRUE Jeans v_c')
ax4.plot(jeans_results['R_grid'], jeans_results['v_circular_grid'], 'k-', 
         linewidth=3, label='Jeans profile')
ax4.set_xlabel('Radial Distance (kpc)')
ax4.set_ylabel('Circular Velocity (km/s)')
ax4.set_title('TRUE Jeans vs Simple Kinematic')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Model comparison
ax5 = axes[1, 1]
if len(jeans_results_models) > 1:
    for model_name, result in jeans_results_models.items():
        ax5.plot(result['R_grid'], result['v_circular_grid'], 
                linewidth=2, label=f'{model_name}')
    ax5.set_xlabel('Radial Distance (kpc)')
    ax5.set_ylabel('Circular Velocity (km/s)')
    ax5.set_title('Density Model Comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
else:
    ax5.text(0.5, 0.5, 'Only one model tested', transform=ax5.transAxes, 
            ha='center', va='center', fontsize=14)

# Plot 6: Jeans correction vs radius
ax6 = axes[1, 2]
velocity_difference = v_circ_jeans - np.abs(v_tang_simple)
ax6.scatter(R_jeans, velocity_difference, alpha=0.3, s=5, color='purple')
ax6.axhline(y=0, color='black', linestyle='--', alpha=0.7)
ax6.set_xlabel('Radial Distance (kpc)')
ax6.set_ylabel('v_Jeans - |v_φ| (km/s)')
ax6.set_title('Jeans Equation Correction')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nTRUE Jeans equation with theoretical density complete!")
print(f"Model used: {jeans_results['density_model']}")
print(f"Mean Jeans correction: {np.mean(velocity_difference):.2f} ± {np.std(velocity_difference):.2f} km/s")
#%%
# Save results
true_jeans_output = pd.DataFrame({
    'radial_distance_kpc': jeans_results['radial_distance'],
    'circular_velocity_jeans_kms': jeans_results['circular_velocity_jeans'],
    'circular_velocity_jeans_err_kms': jeans_results['circular_velocity_jeans_err'],
    'tangential_velocity_kms': jeans_results['tangential_velocity'],
    'stellar_density_theoretical': jeans_results['stellar_density_theoretical'],
    'sigma_radial_kms': jeans_results['sigma_radial'],
    'sigma_tangential_kms': jeans_results['sigma_tangential'],
    'density_model': jeans_results['density_model']
})

#true_jeans_output.to_csv("true_jeans_theoretical_density.csv", index=False)
print(f"Saved results to 'true_jeans_theoretical_density.csv'")