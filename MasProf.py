import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.constants import G, M_sun, R_sun

import scipy.integrate

from BaryonicMatterDesnity import calculate_total_baryonic_mass, calculate_total_mass_with_dm

coeff0=[0, #5th order polynomial
92.7597672696649,
-14.211008443582985,
0.9949471329796322,
-0.03127758534395031,
0.00031983624776082265]

coeff1=[0, #4th order polynomial
    88.87271023314325,
-12.648403056862552,
0.767475837436515,
-0.01708745762667805]

coeff=np.array([
    -69.29164367616694,
132.87563994035452,
-23.013334850165638,
1.9149358241805479,
-0.07724086939671881,
0.0012007573166722646
    ])

coeff_err=np.array([
    102.3158423837855,
60.41414353286173,
13.79547950944647,
1.5099420054149435,
0.07911519954511063,
0.0015893660313373518,
    ])

coeff = coeff[::-1]

coeff_err=coeff_err[::-1]

p1 = np.poly1d(coeff)

p2=np.poly1d(coeff-coeff_err)

p3=np.poly1d(coeff+coeff_err)

#v=root(GM/r)
#M=rv^2/G

kpc_in_meters = u.kpc.to(u.m)
print(G)

#Start from 4-16
#We overestimate mass

r_len=np.linspace(0, 16, 50)
r = r_len * kpc_in_meters
M = p1(r_len)**2 * 1e6 * r/G.value
M2 = p2(r_len)**2 * 1e6 * r/G.value
M3 = p3(r_len)**2 * 1e6 * r/G.value

plt.plot(r_len, M, label="1")
#plt.plot(r_len, M2, label="2")
#plt.plot(r_len, M3, label="3")

plt.legend()
plt.show()

#%%
M_tot = np.zeros(len(r_len))#Total mass with dark matter
M_bar = np.zeros(len(r_len)) #Total Baryonic Mass

for i in range(len(r_len)):
    a=calculate_total_mass_with_dm(r_len[i])
    M_tot[i]=a['total_mass']
    M_bar[i] = a['total_baryonic_mass']
    print(i)

#%%
M = p1(r_len)**2 * 1e6 * r / G.value


#M_dm_obs=M_tot-M_bar  #Test delete later

M_dm_obs=M-M_bar 

#M_dm_S=M_dm_obs//M_sun.to(u.kg).value

#%%

M_s =M/M_sun.to(u.kg).value

plt.plot(r_len, M_s, label="Data")

plt.plot(r_len, M_tot, label="Total Mass with DM")
plt.plot(r_len, M_bar, label="Baryonic Mass")

plt.legend()


plt.show()
#%%

ind=12
max1=47

M_dm_obs=M_dm_obs[ind:max1-1]

r_len=r_len[ind:max1-1]


#%%
from scipy.optimize import curve_fit

def nfw_mass_profile(r, rho_s, r_s):
    # r in kpc, r_s in kpc, rho_s in Msun/kpc^3
    # Output: mass in Msun
    r = np.asarray(r)
    G_const = 4 * np.pi * rho_s * r_s**3
    x = r / r_s
    return G_const * (np.log(1 + x) - x / (1 + x))



# Prepare data for fitting (exclude r=0 and any negative/NaN values)
mask = (r_len > 0.1) & np.isfinite(M_dm_obs) & (M_dm_obs > 0)
r_fit = r_len[mask]
M_dm_fit = M_dm_obs[mask] / M_sun.to(u.kg).value  # Convert to Msun if needed

# Initial guesses: rho_s ~ 0.01 Msun/pc^3 (1e7 Msun/kpc^3), r_s ~ 10 kpc
p0 = [5*1e6, 16]

popt, pcov = curve_fit(nfw_mass_profile, r_fit, M_dm_fit, p0=p0, maxfev=10000)
rho_s_fit, r_s_fit = popt

print(f"Best-fit NFW parameters: rho_s = {rho_s_fit:.2e} Msun/kpc^3, r_s = {r_s_fit:.2f} kpc")

# Plot the fit
r_plot = np.linspace(r_fit.min(), r_fit.max(), 200)
M_nfw_fit = nfw_mass_profile(r_plot, *popt)

plt.plot(r_plot, M_nfw_fit, 'k--', label='NFW Fit')
plt.plot(r_fit, M_dm_fit, 'o', label='Observed DM Mass')
plt.xlabel('Radius (kpc)')
plt.ylabel(r'Dark Matter Mass [$M_\odot$]')
plt.legend()
plt.title('NFW Fit to Dark Matter Mass Profile')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%%

def burkert_mass_profile(r, rho_0, r_c):
    # r in kpc, r_c in kpc, rho_0 in Msun/kpc^3
    # Output: mass in Msun
    r = np.asarray(r)
    x = r / r_c
    factor = np.log(1 + x) + 0.5 * np.log(1 + x**2) - np.arctan(x)
    return (np.pi * rho_0 * r_c**3) * (2 * factor)

# Initial guesses: rho_0 ~ 1e7 Msun/kpc^3, r_c ~ 10 kpc
p0_burkert = [1e7, 10]

popt_burkert, pcov_burkert = curve_fit(burkert_mass_profile, r_fit, M_dm_fit, p0=p0_burkert, maxfev=10000)
rho_0_fit, r_c_fit = popt_burkert

print(f"Best-fit Burkert parameters: rho_0 = {rho_0_fit:.2e} Msun/kpc^3, r_c = {r_c_fit:.2f} kpc")
perr_burkert = np.sqrt(np.diag(pcov_burkert))
print(f"Burkert fit uncertainties: rho_0 = {perr_burkert[0]:.2e} Msun/kpc^3, r_c = {perr_burkert[1]:.2f} kpc")


# Plot the Burkert fit
M_burkert_fit = burkert_mass_profile(r_plot, *popt_burkert)
plt.plot(r_plot, M_burkert_fit, 'g-.', label='Burkert Fit')
plt.plot(r_fit, M_dm_fit, 'o', label='Observed DM Mass')
plt.xlabel('Radius (kpc)')
plt.ylabel(r'Dark Matter Mass [$M_\odot$]')
plt.legend()
plt.title('Burkert Fit to Dark Matter Mass Profile')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()