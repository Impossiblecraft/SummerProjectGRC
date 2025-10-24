# -*- coding: utf-8 -*-
"""
Created on Wed May 14 13:06:29 2025

@author: Rajveer Daga
  parallax > 0 AND
  parallax_over_error > 5 AND
  phot_g_mean_flux_over_error > 10 AND
  (phot_bp_mean_mag - phot_rp_mean_mag) < 0.3 AND
  phot_g_mean_mag < 14
  AND teff_gspphot > 30000
  AND phot_g_mean_flux / POWER(parallax / 1000.0, 2) > 10
"""

import pandas
from astroquery.gaia import Gaia

# ADQL query: select stars with good parallax and magnitude
query = """
SELECT TOP 25000
  source_id,
  ra, ra_error, dec, dec_error,
  parallax, parallax_error, l, b,
  phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
  bp_rp,
  teff_gspphot,
  phot_g_mean_flux / POWER(parallax/1000.0, 2) AS rel_lum_g
FROM gaiadr3.gaia_source
WHERE
    teff_gspphot > 30000 AND
    parallax > 0.01 AND
    ruwe BETWEEN 0.7 AND 1.3 AND
    parallax_over_error > 5
    AND (l BETWEEN 0 AND 360)

ORDER BY rel_lum_g DESC
"""
# filters:
# photo_g_mag filters observed brightness, restricted galactic longitudes and latitudes(last two lines)
# ruwe is for goodness of astrometric data, values close to 1 are best
# other filters are obvious


# Run the query
job = Gaia.launch_job_async(query)
results = job.get_results()

# Convert to Pandas DataFrame
df = results.to_pandas()

# Show the first few rows
print(df.head())

# %%
df.to_csv("OTypes.csv", index=False)
