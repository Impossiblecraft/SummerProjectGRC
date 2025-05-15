# -*- coding: utf-8 -*-
"""
Created on Tue May 13 18:21:25 2025

@author: Rajveer Daga
"""

import pandas
from astroquery.gaia import Gaia

# ADQL query: select stars with good parallax and magnitude
query = """
SELECT TOP 100000
    source_id, ra, ra_error, dec, dec_error, pmra, pmra_error, pmdec, pmdec_error,
    parallax, parallax_error, l, b, 
    radial_velocity, radial_velocity_error, rv_expected_sig_to_noise,
    phot_g_mean_mag
FROM gaiadr3.gaia_source
WHERE
    parallax > 1
    AND parallax_over_error > 20
    AND rv_expected_sig_to_noise > 7
    AND phot_g_mean_mag BETWEEN 8 AND 19
    AND (ruwe BETWEEN 0.9 AND 1.1)
    AND radial_velocity IS NOT NULL
    AND radial_velocity_error IS NOT NULL
    AND (radial_velocity_error / radial_velocity) BETWEEN -0.1 AND 0.1
    AND ((l BETWEEN 0 AND 10) OR (l BETWEEN 350 AND 360) OR (l BETWEEN 170 AND 190))
    AND b BETWEEN -90 AND 90
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

df.to_csv("TestSampleData/SampleData2.csv", index=False)
