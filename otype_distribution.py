import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd


file_path = '/Users/alex/Library/CloudStorage/OneDrive-ImperialCollegeLondon/computing/computing project/OTypes.csv'
df = pd.read_csv(file_path)




# Clean and prepare the data
df_clean = df[df['parallax'] > 0].copy()
df_clean['distance_pc'] = 1000.0 / df_clean['parallax']
df_clean['distance_kpc'] = df_clean['distance_pc'] / 1000.0

# Convert galactic coordinates (l, b in degrees) to radians
l_rad = np.deg2rad(df_clean['l'])
b_rad = np.deg2rad(df_clean['b'])
d = df_clean['distance_kpc']

# Cartesian Galactocentric coordinates (Sun at origin, right-handed system)
df_clean['X'] = d * np.cos(b_rad) * np.cos(l_rad)
df_clean['Y'] = d * np.cos(b_rad) * np.sin(l_rad)
df_clean['Z'] = d * np.sin(b_rad)

# --- Static 3D Plot with Matplotlib ---
fig_static = plt.figure(figsize=(10, 8))
ax_static = fig_static.add_subplot(111, projection='3d')
sc = ax_static.scatter(df_clean['X'], df_clean['Y'], df_clean['Z'],
                       c=df_clean['bp_rp'], cmap='plasma', s=5, alpha=0.7)
ax_static.set_xlabel('X [kpc]')
ax_static.set_ylabel('Y [kpc]')
ax_static.set_zlabel('Z [kpc]')
ax_static.set_title('3D Distribution of O-type Stars (Static)')
plt.colorbar(sc, label='BP - RP Color')
plt.tight_layout()
plt.show()

# --- Interactive 3D Plot with Plotly ---
fig_interactive = go.Figure(data=[go.Scatter3d(
    x=df_clean['X'], y=df_clean['Y'], z=df_clean['Z'],
    mode='markers',
    marker=dict(
        size=3,
        color=df_clean['bp_rp'],  # color by BP-RP
        colorscale='Plasma',
        opacity=0.7,
        colorbar=dict(title='BP - RP')
    )
)])
fig_interactive.update_layout(
    title='Interactive 3D Distribution of O-type Stars',
    scene=dict(
        xaxis_title='X [kpc]',
        yaxis_title='Y [kpc]',
        zaxis_title='Z [kpc]'
    ),
    margin=dict(l=0, r=0, b=0, t=30)
)
fig_interactive.show()



pio.renderers.default = 'browser'  # Forces Plotly to open in browser

fig_interactive.show()  

# --- 2D XY Projection (Galactic Plane) ---
fig_xy, ax_xy = plt.subplots(figsize=(10, 8))
sc = ax_xy.scatter(df_clean['X'], df_clean['Y'],
                   c=df_clean['bp_rp'], cmap='plasma', s=5, alpha=0.7)
ax_xy.set_xlabel('X [kpc]')
ax_xy.set_ylabel('Y [kpc]')
ax_xy.set_title('2D Projection: Galactic Plane (XY)')
ax_xy.axhline(0, color='gray', lw=0.5)
ax_xy.axvline(0, color='gray', lw=0.5)
cbar = plt.colorbar(sc, ax=ax_xy, label='BP - RP Color')
ax_xy.set_aspect('equal')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Assessing Vertical Structure: R vs Z ---
df_clean['R'] = np.sqrt(df_clean['X']**2 + df_clean['Y']**2)

fig_rz, ax_rz = plt.subplots(figsize=(10, 6))
hb = ax_rz.hexbin(df_clean['R'], df_clean['Z'], gridsize=100, cmap='plasma', bins='log')
ax_rz.set_xlabel('Galactocentric Radius R [kpc]')
ax_rz.set_ylabel('Z [kpc]')
ax_rz.set_title('Vertical Disk Structure: Z vs R')
cbar_rz = plt.colorbar(hb, ax=ax_rz, label='log(Star Count)')
plt.tight_layout()
plt.show()

# --- Assessing Radial Density Gradient ---
# Bin stars radially and compute number per ring area
r_bins = np.linspace(0, 12, 60)
df_clean['R_bin'] = pd.cut(df_clean['R'], bins=r_bins)
density_profile = df_clean.groupby('R_bin').size().reset_index(name='count')
density_profile['R_center'] = [interval.mid for interval in density_profile['R_bin']]

# Area of each ring slice (annular area)
r_outer = r_bins[1:]
r_inner = r_bins[:-1]
ring_area = np.pi * (r_outer**2 - r_inner**2)
density_profile['area_kpc2'] = ring_area
density_profile['density'] = density_profile['count'] / density_profile['area_kpc2']

fig_density, ax_density = plt.subplots(figsize=(10, 6))
ax_density.plot(density_profile['R_center'], density_profile['density'], marker='o')
ax_density.set_yscale('log')
ax_density.set_xlabel('Galactocentric Radius R [kpc]')
ax_density.set_ylabel('Stellar Surface Density [stars/kpc²]')
ax_density.set_title('Radial Stellar Density Profile')
plt.grid(True)
plt.tight_layout()
plt.show()
