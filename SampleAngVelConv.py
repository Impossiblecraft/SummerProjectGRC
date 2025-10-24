# -*- coding: utf-8 -*-
"""
Created on Fri May 16 12:17:49 2025

@author: Rajveer Daga
"""

import astropy as ap
import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord, Galactocentric
import astropy.units as u
from astropy.coordinates import CartesianRepresentation, CartesianDifferential
from astropy.coordinates import galactocentric_frame_defaults


# ra	ra_error	dec	dec_error	pmra	pmra_error	pmdec	pmdec_error	parallax	parallax_error	l	b	radial_velocity
ra = 265.62728682859057
dec = -38.89784364238557
pmra = -0.742864315770756
pmdec = -4.449111219275113
parallax = 0.8569495775445937
rv = 25.70886

distance = (1000.0 / parallax) * u.pc  # Convert mas → pc

star = SkyCoord(
    ra=ra * u.deg,
    dec=dec * u.deg,
    distance=distance,
    pm_ra_cosdec=pmra * u.mas/u.yr,
    pm_dec=pmdec * u.mas/u.yr,
    radial_velocity=rv * u.km/u.s
)

# Print Cartesian (ICRS) coordinates
cart_pos = star.cartesian       # 3D position (x, y, z)
cart_vel = star.velocity        # 3D velocity (vx, vy, vz)
print("== ICRS Cartesian Position (pc) ==")
print(f"x = {cart_pos.x:.2f}, y = {cart_pos.y:.2f}, z = {cart_pos.z:.2f}")
print("== ICRS Velocity (km/s) ==")
print(f"vx = {cart_vel.d_x.to(u.km/u.s):.2f}, vy = {cart_vel.d_y.to(u.km/u.s)
      :.2f}, vz = {cart_vel.d_z.to(u.km/u.s):.2f}")
# === STEP 2: Transform to Galactocentric Frame ===
gcoord = star.transform_to(Galactocentric)

# Print Galactocentric position and velocity
gcart_pos = gcoord.cartesian     # Galactocentric position (x, y, z)
gcart_vel = gcoord.velocity      # Galactocentric velocity (vx, vy, vz)

print("\n== Galactocentric Cartesian Position (kpc) ==")
print(f"x = {gcart_pos.x.to(u.kpc):.3f}, y = {
      gcart_pos.y.to(u.kpc):.3f}, z = {gcart_pos.z.to(u.kpc):.3f}")

print("== Galactocentric Velocity (km/s) ==")
print(f"vx = {gcart_vel.d_x.to(u.km/u.s):.2f}, vy = {gcart_vel.d_y.to(u.km/u.s)
      :.2f}, vz = {gcart_vel.d_z.to(u.km/u.s):.2f}")


# Position and velocity vectors
r = np.array([
    gcart_pos.x.to_value(u.kpc),
    gcart_pos.y.to_value(u.kpc),
    gcart_pos.z.to_value(u.kpc)
]) * u.kpc

v = np.array([
    gcart_vel.d_x.to_value(u.km/u.s),
    gcart_vel.d_y.to_value(u.km/u.s),
    gcart_vel.d_z.to_value(u.km/u.s)
]) * (u.km / u.s)

# Angular momentum vector L = r x v
L = np.cross(r, v)  # result is km·kpc/s

# Compute |r|^2
r_squared = np.sum(r**2)  # unit: kpc^2

# Convert L to km·kpc/s explicitly
L = L.to(u.kpc * u.km / u.s)

# Compute angular velocity vector: ω = L / |r|²
omega = L / r_squared  # units: km/s/kpc

# Output
print("\n== Angular Momentum (kpc·km/s) ==")
print(f"Lx = {L[0]:.3f}, Ly = {L[1]:.3f}, Lz = {L[2]:.3f}")

print("\n== Angular Velocity Vector (km/s/kpc) ==")
print(f"ωx = {omega[0]:.5f}, ωy = {omega[1]:.5f}, ωz = {omega[2]:.5f}")

# Angular speed (magnitude)
omega_mag = np.linalg.norm(omega)
print(f"\nAngular Speed |ω| = {omega_mag:.5f} km/s/kpc")
