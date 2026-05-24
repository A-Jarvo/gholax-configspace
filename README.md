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

## Config-Space Likelihood

`gholax/likelihood/nx2pt_configspace.py` implements `Nx2PTCorrelationFunction`,
a config-space extension of the existing `Nx2PTAngularPowerSpectrum` likelihood.
Instead of angular power spectra C_ℓ, it computes real-space correlation functions:

- `w(θ)` — galaxy clustering angular correlation function
- `γ_t(θ)` — galaxy-shear cross-correlation
- `ξ+(θ)`, `ξ-(θ)` — cosmic shear correlation functions

The C_ℓ→ξ(θ) transform uses bin-averaged Legendre kernels following
Schneider et al. (2002) appendix B.

### Validation
Validated against pyccl at a reference ΛCDM cosmology (10-300 arcmin, log-spaced bins):
- w(θ): 0.4% disagreement
- γ_t(θ): 3% disagreement 
- ξ+(θ): 9% agreement (bin-averaging vs point-evaluation difference)
- ξ-(θ): 5% disagreement (known numerical issue with G_neg cancellation at low angles)

Numbers differ from reference due to differences in bin-averaging and point-evaluation, and due to numerical issues at small angles. Dissagreement is most present at small angles due to the bin averaging being more significant at small angles.

### Usage
See `config_configspace_test.yaml` and `generate_dummy_datavector.py` in the
repo root for an example configuration.

**Note:** Full end-to-end sampling requires emulator weights.
