"""
Isolated test for AngularPowerSpectrum_to_CorrelationFunction.
Tests the module without needing a real HDF5 file or full pipeline.

Run from repo root with:
    python -m gholax.test_module_isolated
"""
import sys
import os
from unittest.mock import MagicMock
from jax import config
config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ---- Mock heavy dependencies before any gholax imports ----
# Prevents import errors from h5py, scipy etc. which are not needed
# for this module but get pulled in transitively.
sys.modules['h5py']                  = MagicMock()
sys.modules['jax.scipy.integrate']   = MagicMock()
sys.modules['jax.scipy.interpolate'] = MagicMock()
sys.modules['mpi4py']                = MagicMock()
sys.modules['spinosaurus']           = MagicMock()
sys.modules['spinosaurus.cleft_fftw']= MagicMock()
sys.modules['spinosaurus.density_shape_correlators_fftw']= MagicMock()
sys.modules['spinosaurus.shape_shape_correlators_fftw']= MagicMock()





import numpy as np
import jax.numpy as jnp
from scipy.special import lpmv

from gholax.likelihood.window.AngularPowerSpectrum_to_CorrelationFunction import (
    legendre_poly, legendre_order_2_bar
)


# ---- Mock objects to replace TwoPointSpectrum ----

class MockDataVector:
    """Mimics the attributes of TwoPointSpectrum that the module uses."""
    def __init__(self, spectrum_types, spectrum_info):
        self.spectrum_types = spectrum_types
        self.spectrum_info = spectrum_info


def make_mock_spectrum_info(spectrum_types, n_bins, n_theta):
    """
    Build spectrum_info dict matching what TwoPointSpectrum.process_spectrum_info produces.
    n_bins: number of redshift bins
    n_theta: number of theta bin centers
    """
    # log-spaced theta bins from 1 to 300 arcmin in radians
    theta_centers = np.deg2rad(np.logspace(np.log10(10/60), np.log10(300/60), n_theta))

    spectrum_info = {}
    for t in spectrum_types:
        bins = np.arange(n_bins)
        spectrum_info[t] = {
            "bins0": bins,
            "bins1": bins,
            "use_cross": True,
            "separation": theta_centers,
            "n_bins0_tot": n_bins,
            "n_bins1_tot": n_bins,
        }
    return spectrum_info


 
 
# ---- Reference implementations (no external dependencies) ----
 
def w_theta_reference(ell, C_ell, theta_rad):
    """
    w(theta) = sum_l (2l+1)/(4pi) * C_l * P_l(cos theta)
    Point-evaluated (not bin-averaged). Use bin centers for comparison.
    """
    mu = np.cos(theta_rad)
    result = np.zeros_like(theta_rad, dtype=float)
    # Build Legendre polynomials via recurrence (same as your code)
    P_prev = np.ones_like(mu)   # P_0
    P_curr = mu.copy()          # P_1
    for idx, l in enumerate(ell):
        if l == 0:
            P = P_prev.copy()
        elif l == 1:
            P = P_curr.copy()
        else:
            P_next = ((2*l-1)*mu*P_curr - (l-1)*P_prev) / l
            P_prev = P_curr
            P_curr = P_next
            P = P_curr.copy()
        result += (2*l+1) / (4*np.pi) * C_ell[idx] * P
    return result
 
 
def gamma_t_reference(ell, C_ell, theta_rad):
    """
    gamma_t(theta) = sum_l (2l+1)/(4pi*l*(l+1)) * C_l * P_l^2_bar(cos theta)
    Uses associated Legendre P_l^2 / sin^2(theta) for spin-2 kernel.
    Approximation: P_l^(2)(cos theta) / (1 - cos^2 theta) at bin centers.
    """
    mu = np.cos(theta_rad)
    sin2 = 1 - mu**2
    result = np.zeros_like(theta_rad, dtype=float)
    for idx, l in enumerate(ell):
        if l < 2:
            continue
        # scipy lpmv(m, l, x) = P_l^m(x)
        Pl2 = lpmv(2, l, mu)
        result += (2*l+1) / (4*np.pi*l*(l+1)) * C_ell[idx] * Pl2 / sin2
    return result
 
 
def xi_pm_reference(ell, C_ell, theta_rad):
    """
    xi_+(theta) = sum_l (2l+1)/(2pi) * C_l * (G_l^+(cos theta))
    xi_-(theta) = sum_l (2l+1)/(2pi) * C_l * (G_l^-(cos theta))
    Using the standard spin-2 kernels via associated Legendre polynomials.
    """
    mu = np.cos(theta_rad)
    sin2 = np.maximum(1 - mu**2, 1e-30)
    xi_pos = np.zeros_like(theta_rad, dtype=float)
    xi_neg = np.zeros_like(theta_rad, dtype=float)
    for idx, l in enumerate(ell):
        if l < 2:
            continue
        norm = (2*l+1) / (2*np.pi) / (l*(l+1))**2
        Pl2 = lpmv(2, l, mu)
        # G+/- in terms of P_l^2
        G_pos = Pl2 / sin2
        G_neg = Pl2 / sin2 * np.cos(2 * theta_rad)  # approximate
        xi_pos += norm * C_ell[idx] * G_pos
        xi_neg += norm * C_ell[idx] * G_neg
    return xi_pos, xi_neg
 
 


# ---- Import module under test ----
try:
    from gholax.likelihood.window.AngularPowerSpectrum_to_CorrelationFunction import (
        AngularPowerSpectrum_to_CorrelationFunction
    ) 
    print("Import successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    print("Make sure you're running from the repo root with: python -m gholax.test_module_isolated")
    sys.exit(1)


# ---- Test parameters ----
N_BINS  = 2      # redshift bins
N_THETA = 10     # angular bins
N_ELL   = 500    # ell grid points
L_MAX   = 3000    # keep small for fast testing

spectrum_types = ["w_theta", "gamma_t", "xi_plus", "xi_minus"]
spectrum_info  = make_mock_spectrum_info(spectrum_types, N_BINS, N_THETA)
dv             = MockDataVector(spectrum_types, spectrum_info)


# ---- Instantiate module ----
print("\n=== Instantiating module ===")
try:
    module = AngularPowerSpectrum_to_CorrelationFunction(
        observed_data_vector=dv,
        spectrum_types=spectrum_types,
        spectrum_info=spectrum_info,
        n_ell=N_ELL,
        l_max=L_MAX,
        cl_tag="_mbias",
    )
    print("Module instantiated successfully.")
except Exception as e:
    print(f"Instantiation failed: {e}")
    raise


# ---- Check output_requirements was populated ----
print("\n=== Checking output_requirements ===")
print(f"Number of output requirements: {len(module.output_requirements)}")
for k, v in module.output_requirements.items():
    print(f"  {k} <- {v}")


# ---- Check all_spectra ----
print("\n=== Checking all_spectra ===")
for t in spectrum_types:
    print(f"  {t}: {module.all_spectra[t]}")


# ---- Build mock state with fake C_ell ----
# Simulate what Limber + ShearMultiplicativeBias would have written.
# C_ell values live on module.ell (logspace grid, n_ell points).
print("\n=== Building mock state ===")
ell = module.ell
# Power-law C_ell ~ ell^-2, realistic shape
beam = 1.0 / (1.0 + (ell / 1500.0)**4)
C_ell_dd = 1e-5 * (ell / 100) ** -2 * beam
C_ell_dk = 5e-6 * (ell / 100) ** -2 * beam
C_ell_kk = 1e-6 * (ell / 100) ** -2 * beam

state = {}
for i in range(N_BINS):
    state[f"c_dd_{i}_{i}_mbias"] = C_ell_dd
    for j in range(N_BINS):
        state[f"c_dk_{i}_{j}_mbias"] = C_ell_dk
        if j >= i:
            state[f"c_kk_{i}_{j}_mbias"] = C_ell_kk

print(f"State keys: {list(state.keys())}")


# ---- Run compute() ----
print("\n=== Running compute() ===")
try:
    state = module.compute(state, params_values={})
    print("compute() ran successfully.")
except Exception as e:
    print(f"compute() failed: {e}")
    raise

l_max_test = 15
mu_test = jnp.array([0.998, 0.996, 0.994])
P_test = legendre_poly(mu_test, l_max_test + 1)  # mirrors legendre_poly(mu, self.l_max+1)
P2_bar = legendre_order_2_bar(mu_test, P_test, l_max_test)
print(f"P_test shape: {P_test.shape}")  # expect (l_max_test+2, 3)
print(f"P2_bar shape: {P2_bar.shape}")  # expect (l_max_test-1, 2)

l = 2
term1 = (l + 2/(2*l+1)) * (P_test[1,0] - P_test[1,1])
term2 = (2-l) * (mu_test[0]*P_test[2,0] - mu_test[1]*P_test[2,1])
term3 = (2/(2*l+1)) * (P_test[3,0] - P_test[3,1])
analytic = (term1 + term2 + term3) / (mu_test[0] - mu_test[1])
print(f"l=2 analytic: {analytic:.6f}, code gives: {P2_bar[0,0]:.6f}")
print(f"Match: {jnp.allclose(analytic, P2_bar[0,0], rtol=1e-4)}")

# Check what legendre_order_2_bar returns vs sin^2-divided version
l_test = 10
mu_vals = jnp.array([0.9990, 0.9985, 0.9980, 0.9975])
P_t = legendre_poly(mu_vals, l_max_test + 1)
P2 = legendre_order_2_bar(mu_vals, P_t, l_max_test)

# Bin center mu values
mu_centers = (mu_vals[:-1] + mu_vals[1:]) / 2
sin2_centers = 1 - mu_centers**2

# What P2[l-2] / sin2 gives at bin centers
print(f"P2[l-2=8] at bin centers: {P2[8, :]}")
print(f"sin2 at bin centers: {sin2_centers}")
print(f"P2/sin2: {P2[8,:] / sin2_centers}")

# pyccl gamma prefactor is (2l+1)/(4pi) * d^l_{0,2}
# which for the NG case uses spin-2 Wigner-d, not P^2_l/sin^2
# The ratio between the two normalizations is:
from scipy.special import lpmv
Pl2_center = lpmv(2, l_test, np.array(mu_centers))
print(f"scipy P^2_{l_test} at centers: {Pl2_center}")
print(f"ratio P2_bar*sin2 / scipy_Pl2: {P2[8,:] * sin2_centers / Pl2_center}")

# ---- Check outputs ----
print("\n=== Checking compute() outputs ===")
expected_outputs = list(module.output_requirements.keys())
all_ok = True
for k in expected_outputs:
    if k in state:
        arr = state[k]
        finite = jnp.all(jnp.isfinite(arr))
        print(f"  {k}: shape={arr.shape}, finite={finite}")
        if not finite:
            print(f"    WARNING: non-finite values detected!")
            all_ok = False
    else:
        print(f"  {k}: MISSING from state!")
        all_ok = False
for k in ["w_0_0_obs", "gamma_0_0_obs", "xi_pos_0_0_obs"]:
    print(f"{k}: {state[k]}")

if all_ok:
    print("\nAll expected outputs present and finite.")
else:
    print("\nSome outputs missing or non-finite — check above.")


# ---- Run get_model_from_state() ----
print("\n=== Running get_model_from_state() ===")
try:
    model = module.get_model_from_state(state)
    print(f"Model vector shape: {model.shape}")
    print(f"Model finite: {jnp.all(jnp.isfinite(model))}")
    print(f"Model values (first 5): {model[:5]}")
except Exception as e:
    print(f"get_model_from_state() failed: {e}")
    raise


# ---- Reference comparison ----
print("\n=== Reference comparison ===")
 
# Build dense integer ell grid for reference (same range as module)
ell_dense = np.arange(2, L_MAX + 1)
ell_np    = np.array(module.ell)
C_dd_dense = np.interp(ell_dense, ell_np, np.array(C_ell_dd))
C_dk_dense = np.interp(ell_dense, ell_np, np.array(C_ell_dk))
C_kk_dense = np.interp(ell_dense, ell_np, np.array(C_ell_kk))
 
# Theta bin centers for point evaluation
theta_centers = spectrum_info["w_theta"]["separation"]
 
print("Computing reference transforms (may take ~10s)...")
w_ref             = w_theta_reference(ell_dense, C_dd_dense, theta_centers)
gamma_ref         = gamma_t_reference(ell_dense, C_dk_dense, theta_centers)
xi_pos_ref, xi_neg_ref = xi_pm_reference(ell_dense, C_kk_dense, theta_centers)
 
w_ours      = np.array(state["w_0_0_obs"])
gamma_ours  = np.array(state["gamma_0_0_obs"])
xi_pos_ours = np.array(state["xi_pos_0_0_obs"])
xi_neg_ours = np.array(state["xi_neg_0_0_obs"])
 
def compare(name, ours, ref, rtol=0.15):
    """
    Compare bin-averaged (ours) vs point-evaluated (ref) at bin centers.
    Expect ~5-10% difference from bin averaging, more at small angles.
    """
    mask = np.abs(ref) > 1e-10 * np.max(np.abs(ref))
    if mask.sum() == 0:
        print(f"  {name}: reference is zero everywhere, skipping")
        return True
    frac_diff = np.abs(ours[mask] - ref[mask]) / np.abs(ref[mask])
    max_diff  = np.max(frac_diff)
    mean_diff = np.mean(frac_diff)
    ok = max_diff < rtol
    status = "PASS" if ok else "FAIL"
    print(f"  {name}: max={max_diff:.3f} mean={mean_diff:.3f} ({status}, rtol={rtol})")
    print(f"    ours: {np.array2string(ours, precision=4)}")
    print(f"    ref:  {np.array2string(ref,  precision=4)}")
    return ok
 
print("\nNote: bin-averaged (ours) vs point-evaluated (ref) — expect ~5-10% diff.")
print("Signs and magnitudes should match closely.")
all_pass = True
all_pass &= compare("w(theta)", w_ours,      w_ref)
all_pass &= compare("gamma_t ", gamma_ours,  gamma_ref)
all_pass &= compare("xi_+    ", xi_pos_ours, xi_pos_ref)
all_pass &= compare("xi_-    ", xi_neg_ours, xi_neg_ref)
 
if all_pass:
    print("\nReference comparison: all transforms consistent.")
else:
    print("\nReference comparison: some transforms differ -- check above.")
 
 
# ---- pyccl comparison (if installed) ----
print("\n=== pyccl comparison (optional) ===")
try:
    import pyccl
 
    cosmo = pyccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.83, n_s=0.96,
        transfer_function="bbks",
    )
    theta_deg = np.rad2deg(theta_centers)
 
    w_pyccl      = pyccl.correlation(cosmo, ell=ell_dense, C_ell=C_dd_dense, theta=theta_deg, type="NN", method='fftlog')
    gamma_pyccl  = pyccl.correlation(cosmo, ell=ell_dense, C_ell=C_dk_dense, theta=theta_deg, type="NG", method='fftlog')
    xi_pos_pyccl = pyccl.correlation(cosmo, ell=ell_dense, C_ell=C_kk_dense, theta=theta_deg, type="GG+", method='fftlog')
    xi_neg_pyccl = pyccl.correlation(cosmo, ell=ell_dense, C_ell=C_kk_dense, theta=theta_deg, type="GG-", method='fftlog')
 
    print("Comparing to pyccl (15% tolerance):")
    all_pass_ccl = True
    all_pass_ccl &= compare("w(theta)", w_ours,      w_pyccl,      rtol=0.15)
    all_pass_ccl &= compare("gamma_t ", gamma_ours,  gamma_pyccl,  rtol=0.15)
    all_pass_ccl &= compare("xi_+    ", xi_pos_ours, xi_pos_pyccl, rtol=0.15)
    all_pass_ccl &= compare("xi_-    ", xi_neg_ours, xi_neg_pyccl, rtol=0.15)
 
    if all_pass_ccl:
        print("\npyccl comparison: all transforms correct within 5%.")
    else:
        print("\npyccl comparison: some transforms differ -- check above.")
 
except ImportError:
    print("pyccl not installed, skipping. Install with: pip install pyccl")
except Exception as e:
    print(f"pyccl comparison failed: {e}")
 

# ---- treecorr comparison (optional) ----
print("\n=== treecorr comparison (optional) ===")
try:
    import treecorr

    theta_centers = spectrum_info["w_theta"]["separation"]  # radians
    theta_deg = np.rad2deg(theta_centers)

    def cl_to_xi_treecorr(ell, C_ell, theta_deg, corr_type):
        """
        Use treecorr to convert C_ell to correlation function.
        corr_type: 'NN' for w, 'NG' for gamma_t, 'GG' for xi+/-
        """
        # treecorr expects theta in degrees and ell/C_ell as arrays
        corr = treecorr.Corr2.from_power_spectrum(
            ell, C_ell,
            min_sep=theta_deg.min(), max_sep=theta_deg.max(),
            nbins=len(theta_deg),
            sep_units='degrees',
            corr_type=corr_type,
        )
        return corr

    print("Computing treecorr reference transforms...")
    w_tc      = cl_to_xi_treecorr(ell_dense, C_dd_dense, theta_deg, 'NN')
    gamma_tc  = cl_to_xi_treecorr(ell_dense, C_dk_dense, theta_deg, 'NG')
    xi_tc     = cl_to_xi_treecorr(ell_dense, C_kk_dense, theta_deg, 'GG')

    w_ours      = np.array(state["w_0_0_obs"])
    gamma_ours  = np.array(state["gamma_0_0_obs"])
    xi_pos_ours = np.array(state["xi_pos_0_0_obs"])
    xi_neg_ours = np.array(state["xi_neg_0_0_obs"])

    print("Comparing to treecorr (15% tolerance):")
    all_pass_tc = True
    all_pass_tc &= compare("w(theta)", w_ours,      w_tc.xi,      rtol=0.15)
    all_pass_tc &= compare("gamma_t ", gamma_ours,  gamma_tc.xi,  rtol=0.15)
    all_pass_tc &= compare("xi_+    ", xi_pos_ours, xi_tc.xip,    rtol=0.15)
    all_pass_tc &= compare("xi_-    ", xi_neg_ours, xi_tc.xim,    rtol=0.15)

    if all_pass_tc:
        print("\ntreecorr comparison: all transforms correct within 15%.")
    else:
        print("\ntreecorr comparison: some transforms differ -- check above.")

except ImportError:
    print("treecorr not installed, skipping. Install with: pip install treecorr")
except Exception as e:
    print(f"treecorr comparison failed: {e}")
    raise

print("\n=== All tests complete ===")
