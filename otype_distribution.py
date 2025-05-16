import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd

# Load the uploaded CSV file to examine its structure
file_path = '/Users/alex/Library/CloudStorage/OneDrive-ImperialCollegeLondon/computing/computing project/OTypes.csv'
df = pd.read_csv(file_path)

# Display the first few rows and column names
df.head(), df.columns



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

fig_interactive.show()  # Will now open in Chrome/Firefox/etc.