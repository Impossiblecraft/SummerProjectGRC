import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from astropy.constants import G
import astropy.units as u
import matplotlib.pyplot as plt

data=pd.read_csv("power_series_fit_results_zero_intercept_4bin1.csv")

rad=data['radius_kpc'].values
ob_v=data['observed_velocity_kms'].values
v_err=data['velocity_error_kms'].values

barcoeff=np.loadtxt("baryonic_mass_polyfit_order100.txt")
barfit=np.poly1d(barcoeff)

#MSun=u.Msun.to(u.kg).value

def nfw_mass(r, rho_s, r_s):
    # r in kpc, rho_s in Msun/kpc^3, r_s in kpc
    #G_kpc_kms = G.to(u.kpc * u.km**2 / u.s**2 / u.Msun).value
    x = r / r_s
    mass = 4 * np.pi * rho_s * r_s**3 * (np.log(1 + x) - x / (1 + x))
    #v = np.sqrt(G_kpc_kms * mass / r)
    return mass

def burkert_mass(r, rho_0, r_c):
    # r in kpc, rho_0 in Msun/kpc^3, r_c in kpc
    #G_kpc_kms = G.to(u.kpc * u.km**2 / u.s**2 / u.Msun).value
    x = r / r_c
    mass = (np.pi * rho_0 * r_c**3) * (np.log(1 + x) + 0.5 * np.log(1 + x**2) - np.arctan(x)) * 2
    #v = np.sqrt(G_kpc_kms * mass / r)
    return mass

def rot_velocity_nfw(r, rho_s, r_s):
    # r in kpc, rho_s in Msun/kpc^3, r_s in kpc
    G_kpc_kms = G.to(u.kpc * u.km**2 / u.s**2 / u.Msun).value
    mass = nfw_mass(r, rho_s, r_s)+barfit(r)
    return np.sqrt(G_kpc_kms * mass / r)

def rot_velocity_burkert(r, rho_0, r_c):
    # r in kpc, rho_0 in Msun/kpc^3, r_c in kpc
    G_kpc_kms = G.to(u.kpc * u.km**2 / u.s**2 / u.Msun).value
    mass = burkert_mass(r, rho_0, r_c)+barfit(r)
    return np.sqrt(G_kpc_kms * mass / r)

#pull = 

#%%
# Fit NFW+bar
p0_nfw = [1e7, 10]  # initial guess: rho_s [Msun/kpc^3], r_s [kpc]
popt_nfw, pcov_nfw = curve_fit(
    rot_velocity_nfw, rad, ob_v, sigma=v_err, p0=p0_nfw, absolute_sigma=True, maxfev=10000
)
print(f"NFW best-fit: rho_s = {popt_nfw[0]:.2e} Msun/kpc^3, r_s = {popt_nfw[1]:.2f} kpc")

# Fit Burkert+bar
p0_burkert = [1e7, 10]
popt_burkert, pcov_burkert = curve_fit(
    rot_velocity_burkert, rad, ob_v, sigma=v_err, p0=p0_burkert, absolute_sigma=True, maxfev=10000
)
print(f"Burkert best-fit: rho_0 = {popt_burkert[0]:.2e} Msun/kpc^3, r_c = {popt_burkert[1]:.2f} kpc")

# Calculate chi-squared for both fits
v_nfw_fit = rot_velocity_nfw(rad, *popt_nfw)
v_burkert_fit = rot_velocity_burkert(rad, *popt_burkert)

chi2_nfw = np.sum(((ob_v - v_nfw_fit) / v_err) ** 2)
chi2_burkert = np.sum(((ob_v - v_burkert_fit) / v_err) ** 2)
dof = len(rad) - 2  # 2 fit parameters

print(f"NFW: chi2 = {chi2_nfw:.2f}, reduced chi2 = {chi2_nfw/dof:.2f}")
print(f"Burkert: chi2 = {chi2_burkert:.2f}, reduced chi2 = {chi2_burkert/dof:.2f}")

# Plot the fits
r_plot = np.linspace(0, np.max(rad), 200)
plt.errorbar(rad, ob_v, yerr=v_err, fmt='o', label='Data', alpha=0.5, markersize=2, capsize=5, capthick=1.2)
plt.plot(r_plot, rot_velocity_nfw(r_plot, *popt_nfw), 'r-', label='NFW+Bar')
plt.plot(r_plot, rot_velocity_burkert(r_plot, *popt_burkert), 'g--', label='Burkert+Bar')

# Literature curves
plt.plot(r_plot, rot_velocity_nfw(r_plot, 0.184e7, 16.1), 'b-.', label='NFW (McMillan 2017)')
plt.plot(r_plot, rot_velocity_burkert(r_plot, 1.57e7, 9.26), 'm:', label='Burkert (Sofue 2015)')


plt.xlabel('Radius (kpc)')
plt.ylabel('Rotational Velocity (km/s)')
plt.legend()
plt.title('Rotation Curve Fit: NFW & Burkert (with Baryons)')
#plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#%%


