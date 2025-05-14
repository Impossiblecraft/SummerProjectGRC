from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd

#load data
df = pd.read_csv("SampleData1.csv")  

#Convert to Galactic coordinates
coords = SkyCoord(
    ra=df['ra'].values * u.deg,
    dec=df['dec'].values * u.deg,
    distance=(1 / (df['parallax'].values * u.mas)) * u.parsec,
    frame='icrs'
)



df['l_calc'] = coords.galactic.l.deg
df['b_calc'] = coords.galactic.b.deg

#Filter stars near the Galactic plane (|b| < 5°)
df_filtered = df[abs(df['b']) < 5]

# For now, keep only the useful columns
df_filtered = df_filtered[['source_id', 'ra', 'dec', 'parallax', 'radial_velocity', 'l', 'b']]
df_filtered.to_csv("FilteredData1.csv", index=False)

print(f"Filtered stars saved: {len(df_filtered)} rows")
