"""
Generate a minimal synthetic HDF5 data vector for testing nx2pt_configspace.
Produces a file with w_theta, gamma_t, xi_plus, xi_minus spectra,
Gaussian n(z) for lens and source bins, and a diagonal covariance.

Run with:
    python generate_dummy_datavector.py
Output:
    dummy_configspace_dv.h5
"""
import numpy as np
import h5py

# ---- Configuration ----
N_LENS_BINS   = 2
N_SOURCE_BINS = 2
N_THETA       = 20   # angular bins
N_Z           = 100  # z grid points for n(z)

# Theta bin centers: 10 to 300 arcmin in radians, log-spaced
theta_centers = np.deg2rad(np.logspace(np.log10(10/60), np.log10(300/60), N_THETA))

# Redshift grid
z_grid = np.linspace(0.0, 3.0, N_Z)

# ---- n(z): simple Gaussians ----
def gaussian_nz(z, z_mean, sigma):
    nz = np.exp(-0.5 * ((z - z_mean) / sigma)**2)
    nz /= np.trapz(nz, z)
    return nz

# Lens n(z): two bins at z=0.3, 0.5
lens_means  = [0.3, 0.5]
lens_sigmas = [0.05, 0.05]

# Source n(z): two bins at z=0.7, 1.0
source_means  = [0.7, 1.0]
source_sigmas = [0.1, 0.1]

# Format: first column = z, remaining = n(z) per bin
nz_d = np.zeros((N_Z, N_LENS_BINS + 1))
nz_d[:, 0] = z_grid
for i, (zm, zs) in enumerate(zip(lens_means, lens_sigmas)):
    nz_d[:, i+1] = gaussian_nz(z_grid, zm, zs)

nz_s = np.zeros((N_Z, N_SOURCE_BINS + 1))
nz_s[:, 0] = z_grid
for i, (zm, zs) in enumerate(zip(source_means, source_sigmas)):
    nz_s[:, i+1] = gaussian_nz(z_grid, zm, zs)

# ---- Build spectra array ----
dt = np.dtype([
    ("spectrum_type", "S10"),
    ("zbin0", int),
    ("zbin1", int),
    ("separation", float),
    ("value", float),
])

rows = []

# w_theta: lens auto-correlations only
for i in range(N_LENS_BINS):
    for th in theta_centers:
        rows.append((b"w_theta", i, i, th, 1e-4))

# gamma_t: all lens x source pairs
for i in range(N_LENS_BINS):
    for j in range(N_SOURCE_BINS):
        for th in theta_centers:
            rows.append((b"gamma_t", i, j, th, 1e-5))
    
# Load theory model vector
theory = np.loadtxt('examples/a.txt').flatten()
print(f"theory shape: {theory.shape}")  # should be (120,)
xi_plus_theory  = theory[:60].reshape(3, N_THETA)
xi_minus_theory = theory[60:].reshape(3, N_THETA)

# xi_plus: upper triangle of source bins
pair_idx = 0
for i in range(N_SOURCE_BINS):
    for j in range(i, N_SOURCE_BINS):
        for t_idx, th in enumerate(theta_centers):
            rows.append((b"xi_plus", i, j, th, float(xi_plus_theory[pair_idx, t_idx])))
        pair_idx += 1

# xi_minus: upper triangle of source bins
pair_idx = 0
for i in range(N_SOURCE_BINS):
    for j in range(i, N_SOURCE_BINS):
        for t_idx, th in enumerate(theta_centers):
            rows.append((b"xi_minus", i, j, th, float(xi_minus_theory[pair_idx, t_idx])))
        pair_idx += 1

spectra = np.array(rows, dtype=dt)
n_dv = len(spectra)
print(f"Total data vector length: {n_dv}")

# ---- Diagonal covariance ----
# Simple diagonal with 10% relative errors
cov_dt = np.dtype([
    ("spectrum_type0", "S10"),
    ("spectrum_type1", "S10"),
    ("zbin00", int),
    ("zbin01", int),
    ("zbin10", int),
    ("zbin11", int),
    ("separation0", float),
    ("separation1", float),
    ("value", float),
])
cov = np.zeros((n_dv, n_dv), dtype=cov_dt)
for ii in range(n_dv):
    for jj in range(n_dv):
        cov[ii, jj]["spectrum_type0"] = spectra[ii]["spectrum_type"]
        cov[ii, jj]["spectrum_type1"] = spectra[jj]["spectrum_type"]
        cov[ii, jj]["zbin00"]  = spectra[ii]["zbin0"]
        cov[ii, jj]["zbin01"]  = spectra[ii]["zbin1"]
        cov[ii, jj]["zbin10"]  = spectra[jj]["zbin0"]
        cov[ii, jj]["zbin11"]  = spectra[jj]["zbin1"]
        cov[ii, jj]["separation0"] = spectra[ii]["separation"]
        cov[ii, jj]["separation1"] = spectra[jj]["separation"]
        if ii == jj:
            cov[ii, jj]["value"] = (1.0 * abs(spectra[ii]["value"]))**2 + 1e-30

# ---- Write HDF5 ----
outfile = "dummy_configspace_dv.h5"
with h5py.File(outfile, "w") as f:
    f.create_dataset("spectra",    data=spectra)
    f.create_dataset("covariance", data=cov.flatten())
    f.create_dataset("nz_d",       data=nz_d)
    f.create_dataset("nz_s",       data=nz_s)

print(f"Written: {outfile}")
print(f"  spectra shape:    {spectra.shape}")
print(f"  covariance shape: {cov.shape}")
print(f"  nz_d shape:       {nz_d.shape}")
print(f"  nz_s shape:       {nz_s.shape}")
print(f"\nSpectrum types included:")
for t in [b"w_theta", b"gamma_t", b"xi_plus", b"xi_minus"]:
    n = np.sum(spectra["spectrum_type"] == t)
    print(f"  {t.decode()}: {n} rows")
