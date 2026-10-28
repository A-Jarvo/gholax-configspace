# External-vector parameter recovery for the config-space likelihood

**Branch status:** results, **not for merge**. The corresponding PR contains only the transform fixes and the CAMB pytest.

## The external data vector
- C_ℓ: pyccl (Limber, halofit), shear spectra. Transform: `camb.correlations.cl2corr`, bin-averaged uniformly in cos θ. No gholax code is used to generate it.
- 240 points: w(θ) (2 lens autos), γ_t (4), ξ+ (3), ξ− (3); 20 log-spaced θ bins, 10–300 arcmin.
- **Input cosmology (flat ΛCDM + one massive-neutrino state):** H0 = 67.7, ωb = 0.022447, ωc = 0.11923, A_s = 2.107e-9, n_s = 0.9649, w = −1, Σmν = 0.01 eV (`logmnu = −2` in the gholax config; pyccl uses `mass_split="equal"`).
  Derived (pyccl + CAMB): h = 0.677, Ωm ≈ 0.309, σ8 ≈ 0.820, S8 ≈ 0.833.
- n(z): Gaussians, lenses z = 0.3, 0.5 (σ = 0.05), sources z = 0.7, 1.0 (σ = 0.1). Input bias b1 = 1.5, 1.7.
- **Conventions needed to match gholax:** effective bias b1/D(z) (`scale_by_s8z: true`), magnification slope s = 1.0, `zmin_proj = 0.001` (the example config's 0.1 removes lensing from z < 0.1 and changes ξ+ by 20–30% at large angles).
- Vector comparison (10% errors, inside the scale cuts): χ² = 11.8 for 160 points.

## What is sampled
Only A_s, ωc, b_1_0 and b_1_1 are free. Priors (from the config): A_s uniform on [1.101, 3.101] (×1e-9), ωc uniform on [0.08, 0.16], b_1_0 and b_1_1 uniform on [0.5, 4.0]. Fixed at their input values: ωb, H0, n_s, w, logmnu. All other nuisance parameters (b_2, b_s, counterterms, shot noise, photo-z, shear calibration, IA) are fixed at zero or at the config values.
Truth for the comparison: A_s = 2.107, ωc = 0.11923, b_1_0 = 1.5, b_1_1 = 1.7 (config b1, not the physical bias).

## Runs in this branch
Ten noisy realisations, seeds 1–10 (`chains/pyccl5_seed*`): noise 5%, diagonal covariance 5%, external vector, fixed code.
Setup: 4 NUTS chains, Pathfinder initialisation, dense mass matrix, `adaptive_warmup_max_steps: 400`, `n_steps_min: 600`; w(θ) upper cut 0.055 rad (the ℓ < 10 clamp makes larger angles unreliable).

## Data vector checks
The `.h5` data vectors are not committed, so these checks (made on the files used for the runs) record how they were built:
- **Covariance:** median σ/|value| = 0.050, i.e. 5% as requested (`--frac 0.05`).
- **Noise:** the standard deviation of (value − theory)/σ over the 240 points, averaged over the ten seeds, is 0.984. The expected value is 1.00 ± 0.015, so the injected noise matches the covariance.

## Results (`analysis.txt`)
- std z: 1.05, 0.76, 1.11, 0.75 (A_s, b_1_0, b_1_1, ωc): widths consistent with calibrated within the N = 10 scatter (about ±0.22).
- mean z: −0.60, −0.71, +0.19, +0.24.
- Combined estimate of the ten runs vs input (from `overlay_contours.py --combined`): A_s −0.9%, b_1_0 −0.4%, b_1_1 +0.2%, ωc +0.2%.
- Mean d² = 5.8 (4 expected without a systematic). Seeds 1 and 10 are in the tail (d² = 13.2 and 19.5); seed 10's chains are healthy (R̂ − 1 < 0.005, ESS > 450).
- Hotelling T²: p = 0.018 (all parameters), 0.064 (A_s–ωc pair).

**Interpretation:** the recovered parameters sit within about 1% of the input, with a small, consistent offset pattern. This points to a small systematic difference between gholax and the external calculation, of unidentified origin. Candidates, untested: HEFT vs halofit at small angles, the ℓ < 10 clamp, interpolation from the 200-point ℓ grid, a residual convention difference. A separate noiseless run at 10% errors (not included here) showed the same sign and size of offsets.

## How to read `analysis.txt`
- **z** = (posterior mean − truth) / posterior std, per parameter. **q_sigma** is the same position expressed through the marginal's quantile.
- **Mahalanobis² (d²)** = (mean − truth)ᵀ C⁻¹ (mean − truth), with C the posterior covariance of that run. It is the multivariate z², so it accounts for correlations (A_s and ωc have ρ ≈ −0.98). With a Gaussian, unbiased posterior it follows χ² with 4 degrees of freedom (mean 4).
- **Hotelling T²** tests whether the *mean* z-vector across runs is zero, using the run-to-run scatter, so it detects a bias common to all runs.
- p-values assume Gaussian posteriors; N = 10 has limited power. The "combined" numbers from `overlay_contours.py` use a Gaussian product of the ten posteriors and are approximate.

## Caveats
- **Not a blind test:** the bias convention, `zmin_proj` and the w(θ) cut were set after comparing vectors.
- Sensitive to ~1% shifts at best: "no difference detected at the percent level", not "validated".
- The results hold for this one configuration (two lens and two source bins, diagonal covariance, a single fiducial cosmology).

## Files
- `theory_pyccl.npy`: external 240-point vector
- `chains/pyccl5_seed1..10.*`: samples, warmup parameters, Pathfinder initialisation (restart checkpoints and `.h5` data vectors are not committed)
- `runs_pyccl5.json`: run list and truth values
- `analysis.txt`: console output of `python analyze_runs.py runs_pyccl5.json`, saved by redirecting it to a file (it is not normally written)
- `contour_plot.png`: 68% posteriors of the ten runs with the truth marked, plus the combined estimate from --combined flat
- Scripts, all in the repository root: `make_pyccl_theory.py` (external vector), `make_dv.py` (builds each noisy data file), `run_mock_suite.py` (runs the ten seeds), `analyze_runs.py` (statistics), `overlay_contours.py` (contour plot).

## Reproduce
```bash
python make_pyccl_theory.py --out theory_pyccl.npy          # default b1 = 1.5 1.7
python run_mock_suite.py --config example_configs/config_configspace_emulator.yaml \
    --theory theory_pyccl.npy --n 10 --frac 0.05 --tag pyccl5 --zmin-proj 0.001 --w-theta-max 0.055
python analyze_runs.py runs_pyccl5.json > analysis.txt
python overlay_contours.py runs_pyccl5.json contour_plot.png --combined
```

# gholax
<p align="center">
<img src="gholax_logo.png" alt="drawing" width="400"/>
</p>

Differentiable likelihoods through surrogate models. Our logo represents a neural network surrogate model ([ghola](https://dune.fandom.com/wiki/Ghola)) for a theory model ([velocileptors](https://github.com/sfschen/velocileptors) in this case).

Neural network surrogate models and data from https://arxiv.org/abs/2510.18981 will be made available upon acceptance of this work. For early access, please contact jderose@bnl.gov.

## Installation
### Dependencies
- `h5py` for reading emulator data
- `numpy`, `scipy`
- `mpi4py` for running multiple chains simultaneously
- `jax`, `jaxlib`
- `blackjax` for sampling algorithms
- `optax` for minimization
- `interpax` for theory calculations

### NERSC Installation
At NERSC you can run `sh setup_nersc_env.sh` and this should create a functional conda environment,
that you can activate as follows:

```bash
module load python
mamba activate gholax
```
This is equipped with a jupyter kernel named `gholax` that you can use with NERSC's jupyterlab.

### Local Installation 
Analogously, assuming you have `mamba` installed, you can run `sh setup_env.sh` and it will build a functioning environment with `gholax` installed. 
The environment can then be activated by calling `mamba activate gholax`. 

## Running the tests
From the repository root, with the `gholax` environment activated:

```bash
pytest tests
```

The analytic marginalization (AM) tests are slow; to skip them:

```bash
pytest tests -k "not am_"
```
