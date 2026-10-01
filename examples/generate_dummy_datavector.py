"""
Generate a minimal synthetic HDF5 data vector for testing nx2pt_configspace.
Produces a file with w_theta, gamma_t, xi_plus, xi_minus spectra,
Gaussian n(z) for lens and source bins, and a diagonal covariance.

Run with:
    python examples/generate_dummy_datavector.py
Output:
    dummy_configspace_dv.h5
"""
import numpy as np
import h5py

# ---- Configuration ----
N_LENS_BINS   = 2
N_SOURCE_BINS = 2
N_THETA       = 20   # angular bins
N_Z           = 200  # z grid points for n(z)

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

# Load theory model vector
# Theory prediction for xi+/xi- at reference cosmology:
# As=2.107e-9, omch2=0.11923, ombh2=0.022447, H0=67.7, ns=0.9649, w=-1.0, mnu=0.06eV
# Computed with gholax using CLASS + CLEFT bias model, no IA
# Lens bins: z~0.3, 0.5 (sigma=0.05); Source bins: z~0.7, 1.0 (sigma=0.1)
# Theta range: 10-300 arcmin, 20 log-spaced bins
# Order: xi+ (pairs 00, 01, 11) then xi- (pairs 00, 01, 11), 20 theta bins each
theory = np.array([
     9.60633559e-02, 6.24123273e-02, 4.21581993e-02, 4.37838476e-02,
     2.67221938e-02, 2.54945362e-02, 1.87762461e-02, 1.62043018e-02,
     1.48922123e-02, 1.30174479e-02, 1.02305243e-02, 8.57515443e-03,
     7.10506303e-03, 5.86298260e-03, 4.69135647e-03, 3.73666244e-03,
     2.95410554e-03, 2.27488805e-03, 1.73191106e-03, 1.30131639e-03,
     3.06182487e-03, 1.87661435e-03, 1.27450666e-03, 1.40525116e-03,
     8.35885433e-04, 8.97357018e-04, 7.07125177e-04, 6.12853965e-04,
     5.22383405e-04, 4.32004939e-04, 3.50735742e-04, 2.97156846e-04,
     2.40260649e-04, 1.97588525e-04, 1.55910179e-04, 1.22536327e-04,
     9.56237368e-05, 7.23749773e-05, 5.41821710e-05, 3.99259315e-05,
     5.79545030e-04, 5.66804251e-04, 4.93178389e-04, 3.37115479e-04,
     3.18230212e-04, 2.27399798e-04, 1.92740905e-04, 1.53994510e-04,
     1.21017625e-04, 9.80248533e-05, 8.35327626e-05, 6.81127921e-05,
     5.73409742e-05, 4.77156800e-05, 4.05605311e-05, 3.43213542e-05,
     2.89090347e-05, 2.44500628e-05, 2.05049794e-05, 1.72842595e-05,
     7.14200134e-04, 7.03293905e-04, 6.10807564e-04, 4.15210143e-04,
     3.94193359e-04, 2.80353002e-04, 2.38010174e-04, 1.90210200e-04,
     1.49492582e-04, 1.21245144e-04, 1.03489638e-04, 8.43346817e-05,
     7.10411214e-05, 5.91155290e-05, 5.02730837e-05, 4.25449325e-05,
     3.58302487e-05, 3.03008871e-05, 2.54023538e-05, 2.14041724e-05,
     2.83628234e-04, 3.20678961e-04, 2.68775076e-04, 1.63980132e-04,
     1.73951625e-04, 1.12395167e-04, 1.02881900e-04, 8.85306769e-05,
     7.22110409e-05, 5.98270218e-05, 5.29251888e-05, 4.33501539e-05,
     3.71818014e-05, 3.10338721e-05, 2.65235741e-05, 2.23472826e-05,
     1.85872467e-05, 1.54732086e-05, 1.26667696e-05, 1.03932600e-05,
     5.01018549e-04, 5.70128495e-04, 4.77476043e-04, 2.89228243e-04,
     3.08699364e-04, 1.98919780e-04, 1.83212062e-04, 1.57900235e-04,
     1.28520515e-04, 1.06730972e-04, 9.45163504e-05, 7.74192258e-05,
     6.64291411e-05, 5.54332952e-05, 4.73688717e-05, 3.98867588e-05,
     3.31445526e-05, 2.75607964e-05, 2.25275898e-05, 1.84592392e-05,
     8.22105276e-06, 5.69139893e-06, 4.28465913e-06, 4.27380091e-06,
     3.04553828e-06, 2.91524960e-06, 2.31215089e-06, 1.92414405e-06,
     1.63765185e-06, 1.38698862e-06, 1.12935821e-06, 9.55209340e-07,
     7.81930059e-07, 6.44735070e-07, 5.17275053e-07, 4.12469266e-07,
     3.25968880e-07, 2.52021738e-07, 1.92417407e-07, 1.45375015e-07,
     1.10759260e-05, 7.53131172e-06, 5.67492948e-06, 5.83493285e-06,
     4.06902873e-06, 3.96668314e-06, 3.12416867e-06, 2.60079484e-06,
     2.21682496e-06, 1.87138834e-06, 1.51374933e-06, 1.27796751e-06,
     1.03946121e-06, 8.52917051e-07, 6.78773164e-07, 5.36909672e-07,
     4.20945841e-07, 3.22210744e-07, 2.43533794e-07, 1.81783771e-07,
     1.71424570e-05, 1.14053954e-05, 8.63481466e-06, 9.19460527e-06,
     6.24887304e-06, 6.21745961e-06, 4.84958906e-06, 4.02840745e-06,
     3.43035587e-06, 2.87598798e-06, 2.29917449e-06, 1.92990583e-06,
     1.55090980e-06, 1.25983498e-06, 9.87691914e-07, 7.69596528e-07,
     5.94445757e-07, 4.46512186e-07, 3.30689434e-07, 2.42291571e-07,
     8.82299937e-06, 6.35000482e-06, 5.84888135e-06, 5.47463744e-06,
     4.03412135e-06, 3.64918664e-06, 2.85884560e-06, 2.28817466e-06,
     1.89211253e-06, 1.55422801e-06, 1.24134964e-06, 1.03588597e-06,
     8.48407958e-07, 7.08969603e-07, 5.86547967e-07, 4.90620592e-07,
     4.13629604e-07, 3.48265953e-07, 2.94885804e-07, 2.51085000e-07,
     1.20860817e-05, 8.31343735e-06, 7.69191455e-06, 7.23610511e-06,
     5.19108042e-06, 4.76970312e-06, 3.71958742e-06, 2.97756013e-06,
     2.47353556e-06, 2.03799163e-06, 1.63082520e-06, 1.37094689e-06,
     1.12785266e-06, 9.48148396e-07, 7.87196569e-07, 6.60816552e-07,
     5.58840867e-07, 4.71167278e-07, 3.99187318e-07, 3.39612720e-07,
     1.85892973e-05, 1.20187639e-05, 1.12542137e-05, 1.06905665e-05,
     7.39802582e-06, 6.97363745e-06, 5.41687169e-06, 4.34784247e-06,
     3.64593881e-06, 3.02532171e-06, 2.43340900e-06, 2.07022941e-06,
     1.71633586e-06, 1.45584245e-06, 1.21507108e-06, 1.02470784e-06,
     8.69417566e-07, 7.33171684e-07, 6.20413854e-07, 5.27045573e-07,
])
print(f"theory shape: {theory.shape}")  # should be (120,)
w_theory = theory[0:40].reshape(2, N_THETA)
gamma_theory = theory[40:120].reshape(4, N_THETA)
xi_plus_theory = theory[120:180].reshape(3, N_THETA)
xi_minus_theory = theory[180:240].reshape(3, N_THETA)

# w_theta: lens auto-correlations only
pair_idx = 0
for i in range(N_SOURCE_BINS):
    for t_idx, th in enumerate(theta_centers):
        rows.append((b"w_theta", i, i, th, float(w_theory[pair_idx, t_idx])))
    pair_idx += 1

# gamma_t: all lens x source pairs
pair_idx = 0
for i in range(N_SOURCE_BINS):
    for j in range(N_SOURCE_BINS):
        for t_idx, th in enumerate(theta_centers):
            rows.append((b"gamma_t", i, j, th, float(gamma_theory[pair_idx, t_idx])))
        pair_idx += 1

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
outfile = "examples/dummy_configspace_dv.h5"
with h5py.File(outfile, "w") as f:
    f.create_dataset("spectra",    data=spectra)
    f.create_dataset("covariance", data=cov.flatten())
    f.create_dataset("nz_d",       data=nz_d)
    f.create_dataset("nz_s",       data=nz_s)
    f.create_dataset("nz_d_dk",    data=nz_d)

print(f"Written: {outfile}")
print(f"  spectra shape:    {spectra.shape}")
print(f"  covariance shape: {cov.shape}")
print(f"  nz_d shape:       {nz_d.shape}")
print(f"  nz_s shape:       {nz_s.shape}")
print(f"\nSpectrum types included:")
for t in [b"w_theta", b"gamma_t", b"xi_plus", b"xi_minus"]:
    n = np.sum(spectra["spectrum_type"] == t)
    print(f"  {t.decode()}: {n} rows")
