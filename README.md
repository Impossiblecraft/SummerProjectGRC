# SummerProjectGRC
Using Open Source Astronomical Datasets to estimate Dark Matter Percentage and Distribution in the Milky Way

Required Dependencies:
- AstroPy (Install using: https://docs.astropy.org/en/stable/install.html)
- AstroQuery (Install using: https://astroquery.readthedocs.io/en/latest/)
- Pandas
- NumPy
- SciPy



## **ADQL**
ADQL is largely similar to SQL, adapted for astronomical data. Queries can be passed to astronomical datsets via AstroQuery.

Examples and Helpful links:
- https://www.cosmos.esa.int/web/gaia-users/archive/writing-queries
- https://docs.g-vo.org/adql/notes.pdf


## **Filters**
We are aiming to primarily utilise highly precise data. Filters are implmented for maximum accuracy of astrometric data via restricting certain paramters.
The following represent a summary of the currently applied filters:
- Renormalised Unit Weight Error (ruwe) ~ 1 (Considers ra, dec, and parallax error)
- Parallax Error, less than 5-10%
- Radial Velocity Error ~<10%
- High RV signal to noise ratio
- Moderately bright stars (phot_g_mean_mag - assigns brightness to stars relative to GAIA measurements (lower is brighter), values in an intermediate region used due to possible oversaturation from brighter signals and measurement inaccuracies from fainter signals)

