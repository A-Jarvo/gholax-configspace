"""
Export gholax's own model prediction (the 240-point config-space vector) at a chosen cosmology.

Usage (repo root):
    python export_gholax_theory.py --config example_configs/config_configspace_emulator.yaml --out theory_fid.npy
    python export_gholax_theory.py --config ... --out theory_alt.npy --set As=2.4 omch2=0.12

Parameters are taken from the config: 'value' for fixed ones, 'ref' for sampled ones
(these are the fiducial values used to make the mock), derived ones are evaluated.
The output is NOT scale-masked, so it has all 240 points in the data-vector layout:
w_theta (2 autos), gamma_t (4), xi_plus (3), xi_minus (3), 20 theta bins each.
"""
import argparse
import inspect
import sys

import numpy as np
import yaml
import jax

jax.config.update("jax_default_matmul_precision", "float32")   # as in gholax/cli.py
jax.config.update("jax_numpy_dtype_promotion", "standard")

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--out", default="theory_fid.npy")
ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. As=2.4 omch2=0.12")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
like_cfg = cfg["likelihood"]


def collect(block, out):
    for k, p in block.items():
        if not isinstance(p, dict):
            continue
        if "value" in p:
            out[k] = p["value"]
        elif "ref" in p:
            out[k] = p["ref"]


params = {}
collect(like_cfg.get("params", {}), params)
for name, blk in like_cfg.items():
    if name != "params" and isinstance(blk, dict):
        collect(blk.get("params", {}), params)
for s_ in args.set:                      # overrides first, so derived parameters follow them
    k_, v_ = s_.split("=")
    params[k_] = float(v_)
for blk in [like_cfg.get("params", {})] + [b.get("params", {}) for n, b in like_cfg.items()
                                          if n != "params" and isinstance(b, dict)]:
    for k, p in blk.items():
        if isinstance(p, dict) and "derived" in p:
            f = eval(p["derived"])
            params[k] = f(*[params[a] for a in inspect.signature(f).parameters])
print(f"{len(params)} parameters collected; e.g.",
      {k: params[k] for k in list(params)[:6]})

from gholax.util import Model    # imported after jax config, as in cli.py

model = Model(cfg)


like = model.likelihoods['Nx2PTCorrelationFunction']
params_all = dict(zip(model.param_names, model.prior.reference_values))
if params != params_all:
    print(f"params differ from params_all")
    print(params)
    print(params_all)

vec, state = like.predict_model(params_all, {}, apply_scale_mask=False, return_state=True)
vec = np.asarray(vec)
print("exported vector shape:", vec.shape, " finite:", bool(np.all(np.isfinite(vec))))
if vec.shape != (240,):
    print("WARNING: expected 240 points (w 2x20, gamma 4x20, xi+ 3x20, xi- 3x20); check scale-mask / spectrum_info.")
np.save(args.out, vec)
print("saved", args.out)
