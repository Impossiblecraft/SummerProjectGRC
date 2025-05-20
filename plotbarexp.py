# -*- coding: utf-8 -*-
"""
Created on Tue May 20 15:21:39 2025

@author: Rajveer Daga
"""

from BaryonicMatterDesnity import calculate_total_baryonic_mass, calculate_total_mass_with_dm
# input distance in kpc

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

MSun = 1.988475e30
G = 6.674010551359e-11
kpctom = 3.08567758128e19

d = np.arange(0.2, 20.2, 0.2, dtype=float)  # distance in kpc

totalbm = np.zeros(len(d))
azimuthalv = np.zeros(len(d))  # vel in km/s

for i in range(len(d)):
    totalbm[i] = calculate_total_baryonic_mass(
        d[i])['total_baryonic_mass']*MSun
    azimuthalv[i] = np.sqrt(G*totalbm[i]/(d[i]*kpctom))
    # Convert to km/s
    azimuthalv[i] = azimuthalv[i] / 1000
    print(f"Distance: {d[i]:.1f} kpc, Velocity: {azimuthalv[i]:.2f} km/s")
# %%

# Create pandas DataFrame
df = pd.DataFrame({
    'Distance_kpc': d,
    'TotalBaryonicMass_kg': totalbm,
    'AzimuthalVelocity_kms': azimuthalv
})

# Export to CSV
csv_filename = 'baryonic_mass_rotation_curve.csv'
df.to_csv(csv_filename, index=False)
print(f"\nData exported to {csv_filename}")

# Plot the rotation curve
plt.figure(figsize=(10, 6))
plt.plot(d, azimuthalv, 'b-', linewidth=2)
plt.xlabel('Galactocentric Distance (kpc)')
plt.ylabel('Azimuthal Velocity (km/s)')
plt.title('Rotation Curve from Baryonic Mass Distribution')
plt.grid(True)
plt.savefig('baryonic_rotation_curve.png', dpi=300)
plt.show()
