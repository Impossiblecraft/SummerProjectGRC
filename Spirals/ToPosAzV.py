# -*- coding: utf-8 -*-
"""
Created on Fri May 16 12:43:11 2025

@author: Rajveer Daga
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, Galactocentric
import astropy.units as u
from astropy.coordinates import CartesianRepresentation, CartesianDifferential
from astropy.coordinates import galactocentric_frame_defaults

ab = pd.read_csv("SpiralDat1.csv")

maxl = len(ab)

dat = np.zeros((maxl, 6))  # (px, py, pz, vx, vy, vz)


for i in range(maxl):
    ra = ab.iloc[i, 1]
    dec = ab.iloc[i, 3]
    pmra = ab.iloc[i, 5]
    pmdec = ab.iloc[i, 7]
    parallax = ab.iloc[i, 9]
    rv = ab.iloc[i, 13]

    distance = (1000.0 / parallax) * u.pc  # Convert mas → pc

    star = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        distance=distance,
        pm_ra_cosdec=pmra * u.mas/u.yr,
        pm_dec=pmdec * u.mas/u.yr,
        radial_velocity=rv * u.km/u.s
    )

    # === STEP 2: Transform to Galactocentric Frame ===
    gcoord = star.transform_to(Galactocentric)

    # Print Galactocentric position and velocity
    gcart_pos = gcoord.cartesian     # Galactocentric position (x, y, z)
    gcart_vel = gcoord.velocity

    dat[i, 0] = gcart_pos.x.to_value(u.kpc)
    dat[i, 1] = gcart_pos.y.to_value(u.kpc)
    dat[i, 2] = gcart_pos.z.to_value(u.kpc)

    dat[i, 3] = gcart_vel.d_x.to_value(u.km/u.s)
    dat[i, 4] = gcart_vel.d_y.to_value(u.km/u.s)
    dat[i, 5] = gcart_vel.d_z.to_value(u.km/u.s)
    
    if i%5000==0:
        print(i)

# %%
b = dat[:, :]

plt.plot(b[:, 0], b[:, 4], "*")

plt.show()
# %%


df = pd.DataFrame(dat, columns=['px', 'py', 'pz', 'vx', 'vy', 'vz'])
df.to_csv("SpiralRawCartPos.csv", index=False)


# %%

c = b.copy()

k = 0
while k < len(c):
    if (np.abs(c[k, 2]) < 0.3 or np.abs(c[k, 5]) < 10):
        k = k+1
    else:
        c = np.delete(c, k, axis=0)

# %%

d = np.zeros((len(c), 4)) #(px, py, pz, rotv)
for l in range(len(d)):
    d[l, 0] = c[l,0] #px
    d[l,1] = c[l,1] #py
    d[l,2] = c[l,2] #pz
    cross = np.cross([c[l, 0], c[l, 1], c[l, 2]], [c[l, 3], c[l, 4], c[l, 5]])
    c2 = np.abs(cross)/d[l, 0]
    d[l, 3] = np.sqrt(c2[0]**2+c2[1]**2+c2[2]**2)

# %%
plt.plot(d[:, 0], d[:, 3], "*")
plt.show()

exp = pd.DataFrame(d, columns=['px', 'py', 'pz', 'rotv'])
exp.to_csv("Spiralprotv.csv", index=False)
