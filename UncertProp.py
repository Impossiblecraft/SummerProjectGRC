# -*- coding: utf-8 -*-
"""
Created on Fri May 16 12:43:11 2025

@author: Rajveer Daga
"""
#Use Monte Carlo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, Galactocentric
import astropy.units as u

ab = pd.read_csv("RVData2us.csv")

max_stars = len(ab) #-65000 #Just testing shorter dataset, adjust as needed

# Arrays for data and uncertainties
dat = np.zeros((max_stars, 6))  # (px, py, pz, vx, vy, vz)
dat_err = np.zeros((max_stars, 6))  # uncertainties

# Adjust these column indices based on your CSV structure
ra_err_col = 2
dec_err_col = 4
pmra_err_col = 6
pmdec_err_col = 8
parallax_err_col = 10
rv_err_col = 14


#For uncertanity, we create a raneg fo smaples around each value and then coordinates transform all of them and then 
# calculate the mean and standard deviation of the transformed coordinates and velocities.
# This is a Monte Carlo approach to estimate uncertainties in the final positions and velocities.
# Number of samples for uncertainty estimation
n_samples = 50

for i in range(max_stars):
    # Get data values
    ra = ab.iloc[i, 1]
    dec = ab.iloc[i, 3]
    pmra = ab.iloc[i, 5]
    pmdec = ab.iloc[i, 7]
    parallax = ab.iloc[i, 9]
    rv = ab.iloc[i, 13]
    
    # Get uncertainties
    ra_err = ab.iloc[i, ra_err_col]
    dec_err = ab.iloc[i, dec_err_col]
    pmra_err = ab.iloc[i, pmra_err_col]
    pmdec_err = ab.iloc[i, pmdec_err_col]
    parallax_err = ab.iloc[i, parallax_err_col]
    rv_err = ab.iloc[i, rv_err_col]
    
    # Generate samples using numpy arrays
    ra_samples = np.random.normal(ra, ra_err, n_samples)
    dec_samples = np.random.normal(dec, dec_err, n_samples)
    pmra_samples = np.random.normal(pmra, pmra_err, n_samples)
    pmdec_samples = np.random.normal(pmdec, pmdec_err, n_samples)
    parallax_samples = np.abs(np.random.normal(parallax, parallax_err, n_samples))  # Keep positive
    rv_samples = np.random.normal(rv, rv_err, n_samples)
    
    # Calculate distances
    distance_samples = (1000.0 / parallax_samples) * u.pc
    
    # Create SkyCoord object with all samples at once
    star_samples = SkyCoord(
        ra=ra_samples * u.deg,
        dec=dec_samples * u.deg,
        distance=distance_samples,
        pm_ra_cosdec=pmra_samples * u.mas/u.yr,
        pm_dec=pmdec_samples * u.mas/u.yr,
        radial_velocity=rv_samples * u.km/u.s
    )
    
    # Transform all samples to Galactocentric at once
    gcoord_samples = star_samples.transform_to(Galactocentric)
    gcart_pos_samples = gcoord_samples.cartesian
    gcart_vel_samples = gcoord_samples.velocity
    
    # Extract positions and velocities
    x_samples = gcart_pos_samples.x.to_value(u.kpc)
    y_samples = gcart_pos_samples.y.to_value(u.kpc)
    z_samples = gcart_pos_samples.z.to_value(u.kpc)
    vx_samples = gcart_vel_samples.d_x.to_value(u.km/u.s)
    vy_samples = gcart_vel_samples.d_y.to_value(u.km/u.s)
    vz_samples = gcart_vel_samples.d_z.to_value(u.km/u.s)
    
    # Calculate means and standard deviations
    dat[i, 0] = np.mean(x_samples)
    dat[i, 1] = np.mean(y_samples)
    dat[i, 2] = np.mean(z_samples)
    dat[i, 3] = np.mean(vx_samples)
    dat[i, 4] = np.mean(vy_samples)
    dat[i, 5] = np.mean(vz_samples)
    
    dat_err[i, 0] = np.std(x_samples)
    dat_err[i, 1] = np.std(y_samples)
    dat_err[i, 2] = np.std(z_samples)
    dat_err[i, 3] = np.std(vx_samples)
    dat_err[i, 4] = np.std(vy_samples)
    dat_err[i, 5] = np.std(vz_samples)
    
    if i % 100 == 0:
        print(f"Processed {i}/{max_stars} stars")

# %%
b = dat[:, :]
b_err = dat_err[:, :]

# Plot with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(b[:, 0], b[:, 4], yerr=b_err[:, 4], fmt='*', alpha=0.7, markersize=4)
plt.xlabel('Galactocentric X (kpc)')
plt.ylabel('Galactocentric Vy (km/s)')
plt.title('Galactocentric Positions and Velocities with Uncertainties')
plt.grid(True)
plt.show()

# %%
# Save data with uncertainties
df_with_err = pd.DataFrame({
    'px': dat[:, 0], 'px_err': dat_err[:, 0],
    'py': dat[:, 1], 'py_err': dat_err[:, 1],
    'pz': dat[:, 2], 'pz_err': dat_err[:, 2],
    'vx': dat[:, 3], 'vx_err': dat_err[:, 3],
    'vy': dat[:, 4], 'vy_err': dat_err[:, 4],
    'vz': dat[:, 5], 'vz_err': dat_err[:, 5]
})
df_with_err.to_csv("errrawrotcurve80k1.csv", index=False)

# Save nominal values for compatibility
df = pd.DataFrame(dat, columns=['px', 'py', 'pz', 'vx', 'vy', 'vz'])


# %%
c = b.copy()
c_err = b_err.copy()

# Apply filtering with uncertainties preserved
k = 0
while k < len(c):
    if (np.abs(c[k, 2]) < 0.25 or np.abs(c[k, 5]) < 10):
        k = k+1
    else:
        c = np.delete(c, k, axis=0)
        c_err = np.delete(c_err, k, axis=0)

# %%
d = np.zeros((len(c), 2))
d_err = np.zeros((len(c), 2))

for l in range(len(d)):
    # Calculate radial distance and its uncertainty using linear propagation
    r = np.sqrt(c[l, 0]**2 + c[l, 1]**2 + c[l, 2]**2)
    d[l, 0] = r
    
    # Analytical uncertainty propagation for radial distance
    dr_dx = c[l, 0]/r
    dr_dy = c[l, 1]/r
    dr_dz = c[l, 2]/r
    d_err[l, 0] = np.sqrt((dr_dx * c_err[l, 0])**2 + 
                          (dr_dy * c_err[l, 1])**2 + 
                          (dr_dz * c_err[l, 2])**2)
    
    # Calculate rotational velocity
    cross = np.cross([c[l, 0], c[l, 1], c[l, 2]], [c[l, 3], c[l, 4], c[l, 5]])
    c2 = cross/r  # Don't take absolute value here
    d[l, 1] = np.sqrt(c2[0]**2 + c2[1]**2 + c2[2]**2)
    
    # Simple linear approximation for rotational velocity uncertainty
    # This is approximate but avoids Monte Carlo
    pos_uncertainty = np.sqrt(c_err[l, 0]**2 + c_err[l, 1]**2 + c_err[l, 2]**2)
    vel_uncertainty = np.sqrt(c_err[l, 3]**2 + c_err[l, 4]**2 + c_err[l, 5]**2)
    
    # Rough estimate: uncertainty scales with both position and velocity uncertainties
    relative_pos_err = pos_uncertainty / r
    relative_vel_err = vel_uncertainty / d[l, 1] if d[l, 1] > 0 else 0
    d_err[l, 1] = d[l, 1] * np.sqrt(relative_pos_err**2 + relative_vel_err**2)

# %%
# Plot with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(d[:, 0], d[:, 1], xerr=d_err[:, 0], yerr=d_err[:, 1], 
             fmt='*', alpha=0.7, markersize=4)
plt.xlabel('Radial Distance (kpc)')
plt.ylabel('Rotational Velocity (km/s)')
plt.title('Rotation Curve with Uncertainties')
plt.grid(True)
plt.show()

# Save filtered data with uncertainties
exp_with_err = pd.DataFrame({
    'pos_kpc': d[:, 0], 'pos_err_kpc': d_err[:, 0],
    'rotv_kms': d[:, 1], 'rotv_err_kms': d_err[:, 1]
})
#exp_with_err.to_csv("filterdrv1_with_uncertainties.csv", index=False)

# Save without uncertainties for compatibility
exp = pd.DataFrame(d, columns=['pos(kpc)', 'rotv(km/s)'])
#exp.to_csv("filterdrv1.csv", index=False)

"""
Ignore all code below this point, it's for distance weighting and binning
"""
#%%
#All CSV exports are to be done here 
#Change names as needed

#df.to_csv("rawrotcurvebin80k1.csv", index=False) #Converted p and rot v dta with error in cartesian coordinates

exp_with_err = pd.DataFrame({
    'pos_kpc': d[:, 0], 'pos_err_kpc': d_err[:, 0],
    'rotv_kms': d[:, 1], 'rotv_err_kms': d_err[:, 1]
})
#exp_with_err.to_csv("rotcurvebin80k1.csv", index=False) #Use this data for further analysis with uncertainties