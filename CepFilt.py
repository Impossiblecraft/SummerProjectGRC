# -*- coding: utf-8 -*-
"""
Created on Thu May 15 20:48:18 2025

@author: Rajveer Daga
"""


import pandas
from astroquery.gaia import Gaia

# ADQL query: select stars with good parallax and magnitude
query = """
SELECT
 source_id, ra, ra_error, dec, dec_error, pmra, pmra_error, pmdec, pmdec_error,
    parallax, parallax_error, l, b, 
    radial_velocity, radial_velocity_error,, phot_g_mean_mag
FROM gaiadr3.gaia_source
WHERE source_id IN (
  SELECT source_id FROM gaiadr3.vari_cepheid) 
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
#df.to_csv("RVData2us.csv", index=False)
