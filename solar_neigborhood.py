from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord, Galactocentric, CartesianDifferential
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from scipy.stats import binned_statistic_2d
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter
# # ------------------------
# # Query Gaia DR3 for stars with good parallax and radial velocity
# # ------------------------

# query = """
# SELECT TOP 100000
#     source_id,
#     ra, dec,
#     parallax,
#     pmra, pmdec,
#     radial_velocity
# FROM gaiadr3.gaia_source
# WHERE parallax > 2  -- d < 500 pc
#   AND parallax_over_error > 20
#   AND radial_velocity IS NOT NULL
# """

# job = Gaia.launch_job_async(query)
# results = job.get_results()

# # ------------------------
# # Extract and assign units explicitly
# # ------------------------

# ra = results['ra'].data * u.deg
# dec = results['dec'].data * u.deg
# distance = (1000.0 / results['parallax'].data) * u.pc
# pmra = results['pmra'].data * u.mas/u.yr
# pmdec = results['pmdec'].data * u.mas/u.yr
# rv = results['radial_velocity'].data * u.km/u.s

# # ------------------------
# # Transform to Galactocentric frame
# # ------------------------

# # ICRS to Galactocentric transformation
# c_icrs = SkyCoord(
#     ra=ra,
#     dec=dec,
#     distance=distance,
#     pm_ra_cosdec=pmra,
#     pm_dec=pmdec,
#     radial_velocity=rv,
#     frame='icrs'
# )

# c_galcen = c_icrs.transform_to(Galactocentric())

# z = c_galcen.z.to_value(u.pc)
# vz = c_galcen.v_z.to_value(u.km/u.s)

# # ------------------------
# # Apply KDE for smoother spiral visualization
# # ------------------------

# xy = np.vstack([z, vz])
# kde = gaussian_kde(xy)(xy)

# # Sort values by density for visualization
# idx = kde.argsort()
# z_sorted, vz_sorted, kde_sorted = z[idx], vz[idx], kde[idx]

# # ------------------------
# # Plot z–vz vertical phase-space
# # ------------------------

# plt.figure(figsize=(9, 7))
# sc = plt.scatter(z_sorted, vz_sorted, c=np.log10(kde_sorted), s=1, cmap='viridis', rasterized=True)
# cbar = plt.colorbar(sc)
# cbar.set_label(r'$\log_{10}(\mathrm{Density})$', fontsize=12)
# plt.xlabel(r'$z$ [pc]', fontsize=13)
# plt.ylabel(r'$v_z$ [km/s]', fontsize=13)
# plt.title('Vertical Phase-Space Spiral from Gaia DR3 (<500 pc)', fontsize=14)
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# job = Gaia.launch_job_async(query)
# results = job.get_results()
# df = results.to_pandas()
# from astropy.coordinates import Galactocentric

# coords = SkyCoord(ra=df['ra'].values*u.deg,
#                   dec=df['dec'].values*u.deg,
#                   distance=(1e3 / df['parallax'].values)*u.pc,
#                   pm_ra_cosdec=df['pmra'].values*u.mas/u.yr,
#                   pm_dec=df['pmdec'].values*u.mas/u.yr,
#                   radial_velocity=df['radial_velocity'].values*u.km/u.s,
#                   frame='icrs')

# gcoords = coords.transform_to(Galactocentric())

# # Cylindrical velocities for spiral structure
# vx = gcoords.v_x.value
# vy = gcoords.v_y.value
# vz = gcoords.v_z.value
# vR = -vx  # convention: positive outward
# vPhi = vy

# plt.figure(figsize=(8, 6))
# plt.hist2d(vR, vPhi, bins=300, cmap='plasma', norm='log')
# plt.xlabel('$v_R$ [km/s]')
# plt.ylabel('$v_\\phi$ [km/s]')
# plt.colorbar(label='log(N)')
# plt.title('Velocity Distribution in Solar Neighborhood')
# plt.show()



# -----------------------------
# Query O-type stars in Gaia DR3
# -----------------------------

query = """
SELECT TOP 25000
    source_id,
    ra, dec,
    parallax,
    phot_g_mean_mag,
    bp_rp
FROM gaiadr3.gaia_source
WHERE parallax > 0.01              
  AND parallax_over_error > 5    -- good distances
  AND phot_g_mean_mag < 15      -- luminous
  AND bp_rp < 0.5                 -- very blue = OB candidates
"""

job = Gaia.launch_job_async(query)
results = job.get_results()
#%%
# -----------------------------
# Transform to Galactocentric (x, y, z)
# -----------------------------

ra = results['ra'].data * u.deg
dec = results['dec'].data * u.deg
dist = (1000.0 / results['parallax'].data) * u.pc

skycoord = SkyCoord(ra=ra, dec=dec, distance=dist, frame='icrs')
galcen = skycoord.transform_to(Galactocentric())

x = galcen.x.to_value(u.kpc)+ 8.122
y = galcen.y.to_value(u.kpc)
z = galcen.z.to_value(u.kpc)





plt.figure(figsize=(9, 8))
plt.scatter(x, y, s=1, alpha=0.3)
plt.plot(0, 0, 'ro', label='Sun')
plt.xlabel('x [kpc]')
plt.ylabel('y [kpc]')
plt.legend()
plt.title(' O Stars')
plt.gca().set_aspect('equal')
plt.grid(False)
plt.show()


# Step 1: Mask for 1st quadrant and x,y <= 0.5 kpc
mask = (x > 0) & (y > 0) & (x <= 0.5) & (y <= 0.5)
x1 = x[mask]
y1 = y[mask]

# Step 2: Bin with 0.01–0.05 kpc
bin_size = 0.02# try 0.01 or 0.02 for detail
bins = np.arange(0, 0.5 + bin_size, bin_size)

# 2D histogram
H, xedges, yedges = np.histogram2d(x1, y1, bins=[bins, bins])

# Smooth histogram for spiral-like ridge detection
H_smooth = gaussian_filter(H, sigma=0.5)

# Grid for plotting
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

# Step 3: Plot heatmap and contours
plt.figure(figsize=(10, 9))
plt.imshow(H_smooth.T, origin='lower',
           extent=[0, 0.5, 0, 0.5], cmap='viridis', aspect='equal')
plt.contour(X, Y, H_smooth.T, levels=4, colors='white', linewidths=1.2)

plt.colorbar(label='Smoothed OB Star Density')
plt.xlabel('x [kpc]')
plt.ylabel('y [kpc]')
plt.title('Local OB Star Density (< 0.5 kpc) with Spiral-like Ridges')
plt.grid(False)
plt.tight_layout()
# plt.show()
# -----------------------------
# Apply DBSCAN Clustering
# -----------------------------
coords = np.vstack((x, y)).T
db = DBSCAN(eps=0.5, min_samples=50).fit(coords)
labels = db.labels_

# Remove noise points (label == -1)
mask = labels != -1
x_clust = x[mask]
y_clust = y[mask]

# -----------------------------
# Plot Result
# -----------------------------
plt.figure(figsize=(9, 8))
plt.scatter(x_clust, y_clust, s=1, alpha=0.5, c='blue')
plt.plot(0, 0, 'yo', label='Sun')
plt.xlabel('x [kpc]')
plt.ylabel('y [kpc]')
plt.title('Clustered O Stars in Galactocentric XY Plane')
plt.legend()
plt.gca().set_aspect('equal')
plt.grid(False)
plt.tight_layout()
plt.show()


#%%
#from scipy.spatial.distance import pdist
# -----------------------------
# Spiral Arm Definitions
# -----------------------------
def spiral(R_ref, theta_ref_deg, pitch_deg, dtheta=np.pi, npts=1000):
    """
    Logarithmic spiral arm in Galactocentric coordinates.
    R_ref: reference radius (kpc)
    theta_ref_deg: reference angle (deg)
    pitch_deg: pitch angle (deg)
    dtheta: range to extend on either side of theta_ref (radians)
    npts: number of points
    """
    theta_ref = np.deg2rad(theta_ref_deg)
    pitch = np.deg2rad(pitch_deg)
    # Symmetric theta range around theta_ref
    theta_range = np.linspace(theta_ref - dtheta, theta_ref + dtheta, npts)
    r = R_ref * np.exp((theta_range - theta_ref) * np.tan(pitch))
    # Only keep r > 0
    mask = r > 0
    x_arm = r[mask] * np.cos(theta_range[mask])
    y_arm = r[mask] * np.sin(theta_range[mask])
    return x_arm, y_arm

arms = {
    'Scutum':     (5,  27.6, 19.8),
    'Sagittarius':(6.5,  25.6, 6.9-1.6),
    'Perseus':    (10, 14.2, 9.4-1.4),
    'Cygnus':     (8.4, 8.9, 12.8)
}

# Stack x, y into 2D array for clustering
XY = np.vstack([x, y]).T

# Desired number of stars per bin
stars_per_bin = 200
n_clusters = len(XY) // stars_per_bin

# Perform KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(XY)

# Get cluster centers
centroids = kmeans.cluster_centers_

# Count how many stars in each cluster
counts = np.bincount(labels)

#Removing Stars around sol due to data overdensity
a=0
for i in range(len(counts)):
    if np.abs(centroids[a, 0])<=0.3 and np.abs(centroids[a, 1])<=0.3:
        centroids=np.delete(centroids, a, axis=0)
    else:
        a+=1
        

# Plot the results
plt.figure(figsize=(10, 9))
plt.scatter(x, y, c=labels, s=1, cmap='tab20', alpha=0.4)
plt.scatter(centroids[:, 0], centroids[:, 1], c='k', s=20, marker='+', label='Cluster Centers')
plt.xlabel('x [kpc]')
plt.ylabel('y [kpc]')
plt.title(f'OB Star Clustering: ~{stars_per_bin} Stars per Bin')
plt.plot(0, 0, 'ro', label='Sun')
plt.legend()
plt.gca().set_aspect('equal')
plt.grid(False)
plt.tight_layout()
plt.show()

# # -----------------------------
# Plot: OB Stars + Spiral Arms
# # -----------------------------
plt.figure(figsize=(12, 10))
# plt.scatter(x, y, s=1, alpha=0.3, label='OB Candidates', color='navy')
plt.scatter(centroids[:, 0]+8.122, centroids[:, 1], c='k', s=20, marker='+', label='Cluster Centers')
plt.plot(8.122, 0, 'ro', label='Sun')

# Overlay spiral arms
for name, (R_ref, theta_ref, pitch) in arms.items():
    xs, ys = spiral(R_ref, theta_ref, pitch, dtheta=np.pi/4, npts=2000)
    plt.plot(xs, ys, label=f'{name} Arm')


plt.xlabel('x[kpc]', fontsize=14)
plt.ylabel('y [kpc]', fontsize=14)
plt.title('OB Clusters and Spiral Arms', fontsize=20)
plt.legend(fontsize=14)
plt.gca().set_aspect('equal')
#plt.grid(True)
plt.tight_layout()
plt.savefig('SpiralBinned.png', dpi=900)
plt.show()

"""
# Stack x, y into 2D array for clustering
XY = np.vstack([x, y]).T

# DBSCAN parameters
spatial_eps = 0.05   # Minimum spatial scale for a cluster (kpc)
min_stars = 25      # Minimum number of stars per cluster
max_dist_kpc = 0.15  # Maximum allowed diameter for a cluster (kpc)

# Perform DBSCAN clustering
db = DBSCAN(eps=spatial_eps, min_samples=min_stars).fit(XY)
labels = db.labels_

# Post-process: split clusters that are too large
new_labels = np.full_like(labels, -1)
current_label = 0

for k in np.unique(labels[labels != -1]):
    members = XY[labels == k]
    if len(members) < 2:
        continue
    # Compute max pairwise distance
    max_dist = np.max(pdist(members))
    if max_dist <= max_dist_kpc:
        new_labels[labels == k] = current_label
        current_label += 1
    else:
        # If too large, use Agglomerative Clustering to split
        from sklearn.cluster import AgglomerativeClustering
        n_subclusters = int(np.ceil(max_dist / max_dist_kpc))
        subclust = AgglomerativeClustering(n_clusters=n_subclusters, linkage='ward').fit(members)
        for sub in range(n_subclusters):
            mask = (labels == k)
            mask[mask] = (subclust.labels_ == sub)
            new_labels[mask] = current_label
            current_label += 1

labels = new_labels

# Get cluster centers (mean position of each cluster)
unique_labels = np.unique(labels[labels != -1])
centroids = np.array([XY[labels == k].mean(axis=0) for k in unique_labels])

# Count how many stars in each cluster
counts = np.array([np.sum(labels == k) for k in unique_labels])

# Plot the results
plt.figure(figsize=(10, 9))
plt.scatter(x, y, c=labels, s=1, cmap='tab20', alpha=0.4)
plt.scatter(centroids[:, 0], centroids[:, 1], c='k', s=20, marker='+', label='Cluster Centers')
plt.xlabel('x [kpc]')
plt.ylabel('y [kpc]')
plt.title(f'OB Star Clustering: ≥{min_stars} Stars, {spatial_eps} ≤ Cluster Size ≤ {max_dist_kpc} kpc')
#plt.plot(0, 0, 'ro', label='Sun')
plt.legend()
plt.gca().set_aspect('equal')
plt.grid(False)
plt.tight_layout()
plt.show()
"""