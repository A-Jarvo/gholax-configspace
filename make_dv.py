"""
Build a synthetic configspace data vector from a saved theory vector.

Usage:
    python make_dv.py --theory theory_fid.npy --out example_configs/dv_fid_noiseless.h5
    python make_dv.py --theory theory_fid.npy --seed 3 --out example_configs/dv_fid_seed3.h5

--theory : .npy file with the 240-element theory vector, in the same order as the
           old generate_dummy_datavector.py (w_theta 2x20, gamma_t 4x20, xi_plus 3x20,
           xi_minus 3x20).
--frac   : fractional error per point (default 0.10); covariance is diagonal.
--seed   : if given, add one Gaussian noise realisation drawn from that covariance.
           Omit for noiseless data.

The covariance always uses the TRUE theory values, not the noisy ones.
"""
import argparse
import numpy as np
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("--theory", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frac", type=float, default=0.10)
ap.add_argument("--seed", type=int, default=None)
args = ap.parse_args()

N_LENS_BINS, N_SOURCE_BINS, N_THETA, N_Z = 2, 2, 20, 200
trapz = getattr(np, "trapezoid", None) or np.trapz

theta_centers = np.deg2rad(np.logspace(np.log10(10 / 60), np.log10(300 / 60), N_THETA))
z_grid = np.linspace(0.0, 3.0, N_Z)


def gaussian_nz(z, z_mean, sigma):
    nz = np.exp(-0.5 * ((z - z_mean) / sigma) ** 2)
    return nz / trapz(nz, z)


nz_d = np.zeros((N_Z, N_LENS_BINS + 1))
nz_d[:, 0] = z_grid
for i, (zm, zs) in enumerate(zip([0.3, 0.5], [0.05, 0.05])):
    nz_d[:, i + 1] = gaussian_nz(z_grid, zm, zs)
nz_s = np.zeros((N_Z, N_SOURCE_BINS + 1))
nz_s[:, 0] = z_grid
for i, (zm, zs) in enumerate(zip([0.7, 1.0], [0.1, 0.1])):
    nz_s[:, i + 1] = gaussian_nz(z_grid, zm, zs)

theory = np.load(args.theory)
assert theory.shape == (240,), f"expected 240 elements, got {theory.shape}"
sigma = args.frac * np.abs(theory)

values = theory.copy()
if args.seed is not None:
    values = values + np.random.default_rng(args.seed).normal(0.0, sigma)

dt = np.dtype([("spectrum_type", "S10"), ("zbin0", int), ("zbin1", int),
               ("separation", float), ("value", float)])
rows, k = [], 0
blocks = [
    (b"w_theta", [(i, i) for i in range(N_LENS_BINS)]),
    (b"gamma_t", [(i, j) for i in range(N_LENS_BINS) for j in range(N_SOURCE_BINS)]),
    (b"xi_plus", [(i, j) for i in range(N_SOURCE_BINS) for j in range(i, N_SOURCE_BINS)]),
    (b"xi_minus", [(i, j) for i in range(N_SOURCE_BINS) for j in range(i, N_SOURCE_BINS)]),
]
for name, pairs in blocks:
    for (i, j) in pairs:
        for th in theta_centers:
            rows.append((name, i, j, th, float(values[k])))
            k += 1
assert k == 240
spectra = np.array(rows, dtype=dt)
n_dv = len(spectra)

cov_dt = np.dtype([("spectrum_type0", "S10"), ("spectrum_type1", "S10"),
                   ("zbin00", int), ("zbin01", int), ("zbin10", int), ("zbin11", int),
                   ("separation0", float), ("separation1", float), ("value", float)])
cov = np.zeros((n_dv, n_dv), dtype=cov_dt)
cov["spectrum_type0"] = spectra["spectrum_type"][:, None]
cov["spectrum_type1"] = spectra["spectrum_type"][None, :]
cov["zbin00"] = spectra["zbin0"][:, None]
cov["zbin01"] = spectra["zbin1"][:, None]
cov["zbin10"] = spectra["zbin0"][None, :]
cov["zbin11"] = spectra["zbin1"][None, :]
cov["separation0"] = spectra["separation"][:, None]
cov["separation1"] = spectra["separation"][None, :]
cov["value"] = np.diag(sigma ** 2 + 1e-30)

with h5py.File(args.out, "w") as f:
    f.create_dataset("spectra", data=spectra)
    f.create_dataset("covariance", data=cov.flatten())
    f.create_dataset("nz_d", data=nz_d)
    f.create_dataset("nz_s", data=nz_s)
    f.create_dataset("nz_d_dk", data=nz_d)

kind = "noiseless" if args.seed is None else f"noisy (seed {args.seed})"
print(f"Wrote {args.out}: {n_dv} points, frac={args.frac}, {kind}")
