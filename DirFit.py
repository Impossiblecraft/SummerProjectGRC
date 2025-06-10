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

barcoeff=np.loadtxt("baryonic2_mass_polyfit_order100.txt") #New Mass Func
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
nfw_rho_err, nfw_r_err = np.sqrt(np.diag(pcov_nfw))
print(f"NFW best-fit: rho_s = {popt_nfw[0]:.2e} Msun/kpc^3, r_s = {popt_nfw[1]:.2f} kpc")
print("rho_0 error",nfw_rho_err/10**7, "r_s error", nfw_r_err)


# Fit Burkert+bar
p0_burkert = [1e7, 10]
popt_burkert, pcov_burkert = curve_fit(
    rot_velocity_burkert, rad, ob_v, sigma=v_err, p0=p0_burkert, absolute_sigma=True, maxfev=10000
)
burkert_rho_err, burkert_r_err = np.sqrt(np.diag(pcov_burkert))
print(f"Burkert best-fit: rho_0 = {popt_burkert[0]:.2e} Msun/kpc^3, r_c = {popt_burkert[1]:.2f} kpc")
print("rho_0 error(10^7)",burkert_rho_err/10**7, "r_s error(kpc)", burkert_r_err)

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
plt.plot(r_plot, rot_velocity_nfw(r_plot, 1.82e7, 10.7), 'b-.', label='NFW (Soufe 2015)')
plt.plot(r_plot, rot_velocity_burkert(r_plot, 4e7, 9), 'm:', label='Burkert (Nesti and Salucci, 2013)')

#Bland-Hawthorn and Gerard for Baryonic

plt.xlabel('Radius (kpc)')
plt.ylabel('Rotational Velocity (km/s)')
plt.legend()
plt.title('Rotation Curve Fit: NFW & Burkert (with Baryons)')
#plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#%%
rawdata = pd.read_csv("rotcurvebin80k1.csv")

radii = rawdata['pos_kpc'].values
velocities = rawdata['rotv_kms'].values
radii_err = rawdata['pos_err_kpc'].values
velocities_err = rawdata['rotv_err_kms'].values

# Plot the fits
r_plot = np.linspace(0, np.max(rad), 200)

# Overlay muted scatterplot of unbinned data (no errors)
#rawdata = pd.read_csv("rotcurvebin80k1.csv")
#radii = rawdata['pos_kpc'].values
#velocities = rawdata['rotv_kms'].values
#plt.scatter(radii, velocities, color='gray', alpha=0.1, s=3, label='Unbinned Data')

plt.errorbar(rad, ob_v, yerr=v_err, fmt='o', label='Binned Data', alpha=0.6, markersize=3, capsize=7, capthick=2)
plt.plot(r_plot, rot_velocity_nfw(r_plot, *popt_nfw), 'r-', label='NFW')
plt.plot(r_plot, rot_velocity_burkert(r_plot, *popt_burkert), 'g--', label='Burkert')

# Literature curves
plt.plot(r_plot, rot_velocity_nfw(r_plot, 1.82e7, 10.7), 'b-.', label='NFW (Nesti and Salucci, 2013)')
#plt.plot(r_plot, rot_velocity_burkert(r_plot, 4e7, 9), 'm:', label='Burkert (Nesti and Salucci, 2013)')

#Baryonic Curves (Bland-Hawthorn and Gerard)
G_kpc_kms = G.to(u.kpc * u.km**2 / u.s**2 / u.Msun).value
M_bar_r = barfit(r_plot)
v_bar = np.sqrt(G_kpc_kms * M_bar_r / r_plot)
v_bar[np.isnan(v_bar)] = 0
plt.plot(r_plot, v_bar, color='orange', linestyle='-', linewidth=2, label='Baryonic Matter (Expected)')


plt.xlabel('Radial Distance (kpc)')
plt.ylabel('Rotational Velocity (km/s)')
plt.legend(fontsize=10)
plt.title('Fitted Galactic Rotation Curve: NFW & Burkert')
plt.tight_layout()
plt.savefig("RotationCurveFitBaryonic.png", dpi=1000)
plt.show()

#%%
mask = rawdata['rotv_kms'].values <= 310
radii = rawdata['pos_kpc'].values[mask]
velocities = rawdata['rotv_kms'].values[mask]
radii_err = rawdata['pos_err_kpc'].values[mask]
velocities_err = rawdata['rotv_err_kms'].values[mask]

# Plot the binned data with error bars and scatter plot of unbinned data
plt.figure(figsize=(7,5))
plt.scatter(radii, velocities, color='gray', alpha=0.3, s=5, label='Unbinned Data')
plt.errorbar(rad, ob_v, yerr=v_err, fmt='o', label='Binned Data', alpha=0.7, markersize=4, capsize=7, capthick=2)
plt.xlabel('Radial Distance (kpc)')
plt.ylabel('Rotational Velocity (km/s)')
plt.legend()
plt.title('Galactic Rotation Curve: Binned and Unbinned Data')
plt.tight_layout()
plt.savefig("BinningRotCurve.png", dpi=1200)
plt.show()

#%%
from BaryonicMatterDesnity import calculate_total_mass_with_dm
from scipy.optimize import fsolve

#%%
#NFW
# Constants
rho_crit = 1.37e2  # Msun/kpc^3
delta = 200

# Best-fit NFW parameters
rho_s = popt_nfw[0]     # Msun/kpc^3
r_s = popt_nfw[1]         # kpc

# Errors
rho_s_err = nfw_rho_err  # Msun/kpc^3
r_s_err = nfw_r_err      # kpc

# NFW equation for c200
def nfw_balance(c, rho_s_val):
    lhs = rho_s_val * (np.log(1 + c) - c / (1 + c))
    rhs = (delta / 3.0) * rho_crit * c**3
    return lhs - rhs

# Solve for best c200
c_best = fsolve(nfw_balance, 10, args=(rho_s))[0]

# Estimate error in c200
c_rho_plus  = fsolve(nfw_balance, c_best, args=(rho_s + rho_s_err))[0]
c_rho_minus = fsolve(nfw_balance, c_best, args=(rho_s - rho_s_err))[0]
dc_drho = (c_rho_plus - c_rho_minus) / 2

# r_s affects c through: r200 = c * r_s → c = r200 / r_s
# So c varies → r200 varies
c_rs_plus  = (c_best * r_s) / (r_s + r_s_err)
c_rs_minus = (c_best * r_s) / (r_s - r_s_err)
dc_drs = (c_rs_plus - c_rs_minus) / 2

# Total c error
c_err = np.sqrt(dc_drho**2 + dc_drs**2)

# Compute r200 and its error
r200 = c_best * r_s
r200_err = np.sqrt((r_s * c_err)**2 + (c_best * r_s_err)**2)

print(f"Best-fit c_200 = {c_best:.2f} ± {c_err:.2f}")
print(f"Virial radius r_200 = {r200:.2f} ± {r200_err:.2f} kpc")

#%%
# Number of Monte Carlo samples
n_sample=50
baryonic_masses = []

for i in range(n_sample):
    r_bar=np.random.normal(35, 3)

    #Calculate baryonic mass within r200_sample
    bary_mass = calculate_total_mass_with_dm(r_bar)['total_baryonic_mass']
    baryonic_masses.append(bary_mass)
    print(i)

baryonic_masses = np.array(baryonic_masses)

#%%
n_samples = 20000
# Best-fit NFW parameters and errors
rho_s_best = rho_s
r_s_best = r_s
rho_s_sigma = rho_s_err
r_s_sigma = r_s_err


# Arrays to store results
#baryonic_masses = []
nfw_masses = []

# Virial radius and its error from previous calculation
r200_best = r200
r200_sigma = r200_err

for i in range(n_samples):
    # Sample NFW parameters from normal distributions
    rho_s_sample = np.random.normal(rho_s_best, rho_s_sigma)
    r_s_sample = np.random.normal(r_s_best, r_s_sigma)
    r200_sample = np.random.normal(r200_best, r200_sigma)


    # Calculate NFW mass within r200_sample
    # M_NFW(r) = 4πρ_s r_s^3 [ln(1 + r/r_s) - (r/r_s)/(1 + r/r_s)]
    x = r200_sample / r_s_sample
    nfw_mass = 4 * np.pi * rho_s_sample * r_s_sample**3 * (np.log(1 + x) - x / (1 + x))
    if np.isnan(nfw_mass):
        continue
    nfw_masses.append(nfw_mass)
    if (i%1000)==0:
        print(i)


nfw_masses = np.array(nfw_masses)
#%%
print(f"Enclosed baryonic mass at r200 (mean ± std): {np.mean(baryonic_masses):.2e} ± {np.std(baryonic_masses):.2e} Msun")
print(f"NFW (DM halo) mass at r200 (mean ± std): {np.mean(nfw_masses):.2e} ± {np.std(nfw_masses):.2e} Msun")
Bar_nfw=np.array([np.mean(baryonic_masses), np.std(baryonic_masses)])
Dm_nfw=np.array([np.mean(nfw_masses), np.std(nfw_masses)])
Tot_nfw = Bar_nfw + Dm_nfw

percerr=np.std(nfw_masses)/np.mean(nfw_masses)#+ np.std(baryonic_masses)/np.mean(baryonic_masses)
print(f"Total mass at r200 (mean ± std): {Tot_nfw[0]:.2e} ± {Tot_nfw[1]:.2e} Msun")

pos_err=(Dm_nfw[0]+Dm_nfw[1])/(Tot_nfw[0]+Tot_nfw[1])

neg_err=(Dm_nfw[0]-Dm_nfw[1])/(Tot_nfw[0]-Tot_nfw[1])

print("Total DM perc: ", Dm_nfw[0]/Tot_nfw[0], "+/-", "max", pos_err, "min", neg_err)

val_nfw=Dm_nfw[0]/Tot_nfw[0]

print("up", pos_err-val_nfw)
print("down", neg_err-val_nfw)



#%%

#Calculating Virial Radii
#%%
rho_crit = 1.37e2  # critical density in Msun/kpc^3
delta = 200        # overdensity

n_samples = 20000
rho_0_best = popt_burkert[0]
r_c_best = popt_burkert[1]
rho_0_sigma = burkert_rho_err
r_c_sigma = burkert_r_err

r200_burkert_samples = []
M200_burkert_samples = []

for i in range(n_samples):
    # Sample Burkert parameters
    rho_0_sample = np.random.normal(rho_0_best, rho_0_sigma)
    r_c_sample = np.random.normal(r_c_best, r_c_sigma)

    # Solve for r200 for this sample
    def equation_to_solve_mc(r):
        M_enc = burkert_mass(r, rho_0_sample, r_c_sample)
        mean_density = M_enc / ((4/3) * np.pi * r**3)
        return mean_density - delta * rho_crit

    r200_guess = 100  # kpc
    r200_b_sample = fsolve(equation_to_solve_mc, r200_guess)[0]
    r200_burkert_samples.append(r200_b_sample)

    # Calculate M200 for this sample
    M200_b_sample = burkert_mass(r200_b_sample, rho_0_sample, r_c_sample)
    M200_burkert_samples.append(M200_b_sample)
    if (i%1000)==0:
        print(i)
    

r200_burkert_samples = np.array(r200_burkert_samples)
M200_burkert_samples = np.array(M200_burkert_samples)

print(f"Burkert profile virial radius r_200 = {np.mean(r200_burkert_samples):.2f} ± {np.std(r200_burkert_samples):.2f} kpc")
print(f"Burkert profile virial mass M_200 = {np.mean(M200_burkert_samples):.2e} ± {np.std(M200_burkert_samples):.2e} Msun")

# Calculate the ratio of baryonic mass to total mass
Dm_burk=np.array([np.mean(M200_burkert_samples), np.std(M200_burkert_samples)])
Tot_burk= Bar_nfw + Dm_burk
pos_err_burk=(Dm_burk[0]+Dm_burk[1])/(Tot_burk[0]+Tot_burk[1])
neg_err_burk=(Dm_burk[0]-Dm_burk[1])/(Tot_burk[0]-Tot_burk[1])
print("Total DM perc: ", Dm_burk[0]/Tot_burk[0], "+/-", "max", pos_err_burk, "min", neg_err_burk)

val_burk=Dm_burk[0]/Tot_burk[0]

print("up", pos_err_burk-val_burk)
print("down", neg_err_burk-val_burk)
