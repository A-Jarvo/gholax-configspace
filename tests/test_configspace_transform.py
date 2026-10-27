"""
Isolated pytest compatible test for AngularPowerSpectrum_to_CorrelationFunction.
Tests the C_ell -> correlation function transform without needing
a real HDF5 file or full pipeline. Validates agaist camb, pyccl is optional.
"""

import pytest
import sys
import os
from jax import config
from tests.conftest import W0WA_COSMO_PARAMS, requires_classy
config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ---- Mock heavy dependencies before any gholax imports ----
import pytest
from unittest.mock import MagicMock

@pytest.fixture(autouse=True)
def mock_heavy_deps(monkeypatch):
    """Mock heavy dependencies that aren't needed for transform testing."""
    mocks = {
        'h5py': MagicMock(),
        'jax.scipy.integrate': MagicMock(),
        'jax.scipy.interpolate': MagicMock(),
        'mpi4py': MagicMock(),
        'spinosaurus': MagicMock(),
        'spinosaurus.cleft_fftw': MagicMock(),
        'spinosaurus.density_shape_correlators_fftw': MagicMock(),
        'spinosaurus.shape_shape_correlators_fftw': MagicMock(),
    }
    for mod_name, mock in mocks.items():
        monkeypatch.setitem(sys.modules, mod_name, mock)

import numpy as np
import jax.numpy as jnp


# ---- Mock objects to replace TwoPointSpectrum ----

class MockDataVector:
    """Mimics the attributes of TwoPointSpectrum that the module uses."""
    def __init__(self, spectrum_types, spectrum_info):
        self.spectrum_types = spectrum_types
        self.spectrum_info = spectrum_info


def make_mock_spectrum_info(spectrum_types, n_bins, n_theta):
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

from gholax.likelihood.window.AngularPowerSpectrum_to_CorrelationFunction import (
        AngularPowerSpectrum_to_CorrelationFunction)

try:
    import pyccl
    pyccl_imported = True
except ImportError:
    pyccl_imported = False
    
@requires_classy
@pytest.mark.skipif(not pyccl_imported, reason="Requires pyccl")
def test_transform_agaist_camb(mock_heavy_deps):
    N_BINS  = 2
    N_THETA = 10
    N_ELL   = 500
    L_MAX   = 3000
    N_GL    = 96      # Gauss nodes per theta bin for the reference average
    RTOL    = 1e-6    # CAMB reference is exact for the same truncated sum
    CONVERT_SHEAR_TO_KAPPA = True

    
    spectrum_types = ["w_theta", "gamma_t", "xi_plus", "xi_minus"]
    spectrum_info  = make_mock_spectrum_info(spectrum_types, N_BINS, N_THETA)
    dv             = MockDataVector(spectrum_types, spectrum_info)
    module = AngularPowerSpectrum_to_CorrelationFunction(
        observed_data_vector=dv,
        spectrum_types=spectrum_types,
        spectrum_info=spectrum_info,
        n_ell=N_ELL,
        l_max=L_MAX,
        cl_tag="_mbias",
    )
    ell_module = np.array(module.ell)

    # construct power spectra (pyccl)

    # Get standard params from conftest
    params = W0WA_COSMO_PARAMS["wa_zero"]

    H0 = params["H0"]
    ombh2 = params["ombh2"]
    omch2 = params["omch2"]
    mnu = params["mnu"]
    As = params["As"]
    ns = params["ns"]
    w = params["w"]
    wa = params["wa"]

    # transform params to work with pyccl
    h = H0 / 100.0
    h2 = h**2
    omega_b = ombh2 / h2
    omega_c = omch2 / h2
    A_s_scaled = As * 1e-9

    cosmo = pyccl.Cosmology(
        h=h,
        Omega_b=omega_b,
        Omega_c=omega_c,
        m_nu=mnu,
        A_s=A_s_scaled,
        n_s=ns,
        w0=w,
        wa=wa,
        transfer_function="boltzmann_class"
    )
    z = np.linspace(0, 3, 100)

    # Source n(z): Gaussian at z=0.7
    nz_source = np.exp(-0.5*((z - 0.7)/0.1)**2)
    source_tracer = pyccl.WeakLensingTracer(cosmo, dndz=(z, nz_source))

    # Lens n(z): Gaussian at z=0.5, linear bias=1
    nz_lens = np.exp(-0.5*((z - 0.5)/0.1)**2)
    lens_tracer = pyccl.NumberCountsTracer(
        cosmo, has_rsd=False, dndz=(z, nz_lens), bias=(z, np.ones_like(z))
    )

    C_ell_dd = pyccl.angular_cl(cosmo, lens_tracer,   lens_tracer,   ell_module)
    C_ell_dk = pyccl.angular_cl(cosmo, lens_tracer,   source_tracer, ell_module)
    C_ell_kk = pyccl.angular_cl(cosmo, source_tracer, source_tracer, ell_module)

    

    # ---- Shear -> convergence-style conversion (what the kernels expect) ----
    def _f(l):
        return np.sqrt((l + 2) * (l + 1) * l * (l - 1))
    
    if CONVERT_SHEAR_TO_KAPPA:
        print("Converting shear C_ells to convergence-style input for the module.")
        C_ell_dk_in = C_ell_dk * ell_module * (ell_module + 1) / _f(ell_module)
        C_ell_kk_in = C_ell_kk * (ell_module * (ell_module + 1)) ** 2 / _f(ell_module) ** 2
    else:
        print("Feeding raw shear C_ells to the module (no conversion).")
        C_ell_dk_in, C_ell_kk_in = C_ell_dk, C_ell_kk
    

    # build mock statee
    n_bins = N_BINS
    state = {}
    state["c_dd_mbias"] = jnp.stack([jnp.array(C_ell_dd)]    * (n_bins * n_bins))
    state["c_dk_mbias"] = jnp.stack([jnp.array(C_ell_dk_in)] * (n_bins * n_bins))
    state["c_kk_mbias"] = jnp.stack([jnp.array(C_ell_kk_in)] * (n_bins * n_bins))

    state = module.compute(state, params_values={})

    # checking all values are finite and present
    expected_outputs = list(module.output_requirements.keys())
    all_finite = True
    all_present = True
    for k in expected_outputs:
        if k in state:
            arr = state[k]
            finite = jnp.all(jnp.isfinite(arr))
            if not finite:
                all_finite = False
        else:
            all_present = False

    assert all_present, "Some outputs missing"
    assert all_finite, "Some outputs non-finite"

    try:
        from camb import correlations
        camb_available = True
    except ImportError:
        camb_available = False

    if not camb_available:
        pytest.skip("camb not available")
    else:
        
        ell_dense  = np.arange(2, L_MAX + 1)
        C_dd_dense = np.interp(ell_dense, ell_module, np.array(C_ell_dd))
        C_dk_in_dense = np.interp(ell_dense, ell_module, np.array(C_ell_dk_in))
        C_kk_in_dense = np.interp(ell_dense, ell_module, np.array(C_ell_kk_in))
        if CONVERT_SHEAR_TO_KAPPA:   # undo the conversion exactly at integer ell -> shear spectra for CAMB
            C_dk_dense = C_dk_in_dense * _f(ell_dense) / (ell_dense * (ell_dense + 1))
            C_kk_dense = C_kk_in_dense * _f(ell_dense) ** 2 / (ell_dense * (ell_dense + 1)) ** 2
        else:
            C_dk_dense, C_kk_dense = C_dk_in_dense, C_kk_in_dense
    
        theta_centers = spectrum_info["w_theta"]["separation"]
        half_gaps = np.diff(theta_centers) / 2
        theta_edges = np.concatenate([
            [theta_centers[0] - half_gaps[0]],
            theta_centers[:-1] + half_gaps,
            [theta_centers[-1] + half_gaps[-1]]
        ])
    
        mu_edges = np.clip(np.cos(theta_edges), -1 + 1e-6, 1 - 1e-6)   # as in compute()
    
        # CAMB inputs: columns TT, EE, BB, TE with l(l+1)/2pi included
        cls = np.zeros((L_MAX + 1, 4))
        fac = ell_dense * (ell_dense + 1) / (2 * np.pi)
        cls[2:, 0] = C_dd_dense * fac     # TT -> w
        cls[2:, 1] = C_kk_dense * fac     # EE -> xi+/xi-
        cls[2:, 3] = C_dk_dense * fac     # TE -> gamma_t
    
        xg, wg = np.polynomial.legendre.leggauss(N_GL)
        n_theta_bins = len(mu_edges) - 1
        nodes = np.zeros((n_theta_bins, N_GL))
        for b in range(n_theta_bins):
            lo, hi = mu_edges[b + 1], mu_edges[b]            # mu decreases with theta
            nodes[b] = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        corr = correlations.cl2corr(cls, nodes.ravel(), lmax=L_MAX)   # columns: T, Q+U, Q-U, cross
    
        def bin_average(v):
            return (v.reshape(n_theta_bins, N_GL) * (wg / 2.0)).sum(1)
    
        w_ref      = bin_average(corr[:, 0])
        xi_pos_ref = bin_average(corr[:, 1])
        xi_neg_ref = bin_average(corr[:, 2])
        gamma_ref  = bin_average(corr[:, 3])
    
        w_ours      = np.array(state["w_0_0_obs"])
        gamma_ours  = np.array(state["gamma_0_0_obs"])
        xi_pos_ours = np.array(state["xi_pos_0_0_obs"])
        xi_neg_ours = np.array(state["xi_neg_0_0_obs"])

    def compare(name, ours, ref, rtol=RTOL, atol_fraction=0.0):
        """atol_fraction: ignore bins where |ref| < atol_fraction * max(|ref|) (0 = compare all bins)"""
        mask = np.abs(ref) > atol_fraction * np.max(np.abs(ref))
        if mask.sum() == 0:
            print(f"  {name}: reference is zero everywhere, skipping")
            return True
        frac_diff = np.abs(ours[mask] - ref[mask]) / np.abs(ref[mask])
        max_diff  = np.max(frac_diff)
        mean_diff = np.mean(frac_diff)
        ok = max_diff < rtol
        return ok

    all_pass_ref = True
    all_pass_ref &= compare("w(theta)", w_ours,      w_ref)
    all_pass_ref &= compare("gamma_t ", gamma_ours,  gamma_ref)
    all_pass_ref &= compare("xi_+    ", xi_pos_ours, xi_pos_ref)
    all_pass_ref &= compare("xi_-    ", xi_neg_ours, xi_neg_ref)

    assert all_pass_ref, "some transforms differed from camb reference"

@pytest.mark.skipif(pyccl_imported, reason="skipped cause redundant with pyccl")
def test_transform_no_reference(mock_heavy_deps):
    N_BINS  = 2
    N_THETA = 10
    N_ELL   = 500
    L_MAX   = 3000
    N_GL    = 96      # Gauss nodes per theta bin for the reference average
    RTOL    = 1e-6    # CAMB reference is exact for the same truncated sum
    CONVERT_SHEAR_TO_KAPPA = True

    
    spectrum_types = ["w_theta", "gamma_t", "xi_plus", "xi_minus"]
    spectrum_info  = make_mock_spectrum_info(spectrum_types, N_BINS, N_THETA)
    dv             = MockDataVector(spectrum_types, spectrum_info)
    module = AngularPowerSpectrum_to_CorrelationFunction(
        observed_data_vector=dv,
        spectrum_types=spectrum_types,
        spectrum_info=spectrum_info,
        n_ell=N_ELL,
        l_max=L_MAX,
        cl_tag="_mbias",
    )
    ell_module = np.array(module.ell)

    # construct power spectra (power law)

    beam = 1.0 / (1.0 + (ell_module / 1500.0)**4)
    C_ell_dd = 1e-5 * (ell_module / 100) ** -2 * beam
    C_ell_dk = 5e-6 * (ell_module / 100) ** -2 * beam
    C_ell_kk = 1e-6 * (ell_module / 100) ** -2 * beam

    

    # ---- Shear -> convergence-style conversion (what the kernels expect) ----
    def _f(l):
        return np.sqrt((l + 2) * (l + 1) * l * (l - 1))
    
    if CONVERT_SHEAR_TO_KAPPA:
        print("Converting shear C_ells to convergence-style input for the module.")
        C_ell_dk_in = C_ell_dk * ell_module * (ell_module + 1) / _f(ell_module)
        C_ell_kk_in = C_ell_kk * (ell_module * (ell_module + 1)) ** 2 / _f(ell_module) ** 2
    else:
        print("Feeding raw shear C_ells to the module (no conversion).")
        C_ell_dk_in, C_ell_kk_in = C_ell_dk, C_ell_kk
    

    # build mock statee
    n_bins = N_BINS
    state = {}
    state["c_dd_mbias"] = jnp.stack([jnp.array(C_ell_dd)]    * (n_bins * n_bins))
    state["c_dk_mbias"] = jnp.stack([jnp.array(C_ell_dk_in)] * (n_bins * n_bins))
    state["c_kk_mbias"] = jnp.stack([jnp.array(C_ell_kk_in)] * (n_bins * n_bins))

    state = module.compute(state, params_values={})

    # checking all values are finite and present
    expected_outputs = list(module.output_requirements.keys())
    all_finite = True
    all_present = True
    for k in expected_outputs:
        if k in state:
            arr = state[k]
            finite = jnp.all(jnp.isfinite(arr))
            if not finite:
                all_finite = False
        else:
            all_present = False

    assert all_present, "Some outputs missing"
    assert all_finite, "Some outputs non-finite"

    try:
        from camb import correlations
        camb_available = True
    except ImportError:
        camb_available = False

    if not camb_available:
        pytest.skip("camb not available")
    else:
        
        ell_dense  = np.arange(2, L_MAX + 1)
        C_dd_dense = np.interp(ell_dense, ell_module, np.array(C_ell_dd))
        C_dk_in_dense = np.interp(ell_dense, ell_module, np.array(C_ell_dk_in))
        C_kk_in_dense = np.interp(ell_dense, ell_module, np.array(C_ell_kk_in))
        if CONVERT_SHEAR_TO_KAPPA:   # undo the conversion exactly at integer ell -> shear spectra for CAMB
            C_dk_dense = C_dk_in_dense * _f(ell_dense) / (ell_dense * (ell_dense + 1))
            C_kk_dense = C_kk_in_dense * _f(ell_dense) ** 2 / (ell_dense * (ell_dense + 1)) ** 2
        else:
            C_dk_dense, C_kk_dense = C_dk_in_dense, C_kk_in_dense
    
        theta_centers = spectrum_info["w_theta"]["separation"]
        half_gaps = np.diff(theta_centers) / 2
        theta_edges = np.concatenate([
            [theta_centers[0] - half_gaps[0]],
            theta_centers[:-1] + half_gaps,
            [theta_centers[-1] + half_gaps[-1]]
        ])
    
        mu_edges = np.clip(np.cos(theta_edges), -1 + 1e-6, 1 - 1e-6)   # as in compute()
    
        # CAMB inputs: columns TT, EE, BB, TE with l(l+1)/2pi included
        cls = np.zeros((L_MAX + 1, 4))
        fac = ell_dense * (ell_dense + 1) / (2 * np.pi)
        cls[2:, 0] = C_dd_dense * fac     # TT -> w
        cls[2:, 1] = C_kk_dense * fac     # EE -> xi+/xi-
        cls[2:, 3] = C_dk_dense * fac     # TE -> gamma_t
    
        xg, wg = np.polynomial.legendre.leggauss(N_GL)
        n_theta_bins = len(mu_edges) - 1
        nodes = np.zeros((n_theta_bins, N_GL))
        for b in range(n_theta_bins):
            lo, hi = mu_edges[b + 1], mu_edges[b]            # mu decreases with theta
            nodes[b] = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        corr = correlations.cl2corr(cls, nodes.ravel(), lmax=L_MAX)   # columns: T, Q+U, Q-U, cross
    
        def bin_average(v):
            return (v.reshape(n_theta_bins, N_GL) * (wg / 2.0)).sum(1)
    
        w_ref      = bin_average(corr[:, 0])
        xi_pos_ref = bin_average(corr[:, 1])
        xi_neg_ref = bin_average(corr[:, 2])
        gamma_ref  = bin_average(corr[:, 3])
    
        w_ours      = np.array(state["w_0_0_obs"])
        gamma_ours  = np.array(state["gamma_0_0_obs"])
        xi_pos_ours = np.array(state["xi_pos_0_0_obs"])
        xi_neg_ours = np.array(state["xi_neg_0_0_obs"])

    def compare(name, ours, ref, rtol=RTOL, atol_fraction=0.0):
        """atol_fraction: ignore bins where |ref| < atol_fraction * max(|ref|) (0 = compare all bins)"""
        mask = np.abs(ref) > atol_fraction * np.max(np.abs(ref))
        if mask.sum() == 0:
            print(f"  {name}: reference is zero everywhere, skipping")
            return True
        frac_diff = np.abs(ours[mask] - ref[mask]) / np.abs(ref[mask])
        max_diff  = np.max(frac_diff)
        mean_diff = np.mean(frac_diff)
        ok = max_diff < rtol
        return ok

    all_pass_ref = True
    all_pass_ref &= compare("w(theta)", w_ours,      w_ref)
    all_pass_ref &= compare("gamma_t ", gamma_ours,  gamma_ref)
    all_pass_ref &= compare("xi_+    ", xi_pos_ours, xi_pos_ref)
    all_pass_ref &= compare("xi_-    ", xi_neg_ours, xi_neg_ref)

    assert all_pass_ref, "some transforms differed from camb reference"

