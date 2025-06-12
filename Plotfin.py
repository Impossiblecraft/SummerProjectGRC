import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("rotcurvebin80k1.csv")

radii = data['pos_kpc'].values
velocities = data['rotv_kms'].values
radii_err = data['pos_err_kpc'].values
velocities_err = data['rotv_err_kms'].values


# Plotting the rotation curve
a=0
for i in range(len(velocities)):
    if velocities[a]>340 or velocities_err[a]>100:
        radii=np.delete(radii, a)
        velocities=np.delete(velocities, a)
        radii_err=np.delete(radii_err, a)
        velocities_err=np.delete(velocities_err, a)
    else:
        a=a+1

plt.figure(figsize=(10, 6))
plt.errorbar(radii, velocities, xerr=radii_err, yerr=velocities_err,
             fmt='.', alpha=0.1, markersize=1, capsize=0.1, label='Data with Uncertainties', color='blue')
plt.xlabel('Radial Distance (kpc)', fontsize=16)
plt.ylabel('Rotational Velocity (km/s)', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.title('Original Data with Uncertainties', fontsize=20)
plt.savefig("RawScatter.png")
plt.grid(True, alpha=0.3)
plt.show()

