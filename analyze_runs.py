"""
Report parameter recovery in sigma units for one or more chain runs.

Usage:
    python analyze_runs.py runs.json

runs.json is a list like:
[
  {"label": "fid_seed1",
   "prefix": "chains/fid_seed1",
   "truth": {"As": 2.107, "omch2": 0.11923, "b_1_0": 1.5, "b_1_1": 1.7}},
  ...
]

Per run, for each parameter:
  z         = (posterior mean - truth) / posterior std
  q_sigma   = Gaussian-equivalent sigma of the truth's position in the 1D marginal
              (two-sided, from the fraction of samples below the truth)
Per run, jointly:
  Mahalanobis distance of truth from posterior mean (full sample covariance),
  converted to a p-value (chi2, k dof) and a Gaussian-equivalent sigma.
Across runs (needs >= 3): mean and std of z per parameter.
  Unbiased pipeline  ->  mean z ~ 0 (within 1/sqrt(N)),  std z ~ 1.
"""
import glob
import json
import sys

import numpy as np
from scipy import stats

runs = json.load(open(sys.argv[1]))


def load(prefix):
    files = sorted(glob.glob(f"{prefix}.samples.0.*.txt"))
    if not files:
        raise FileNotFoundError(prefix)
    with open(files[0]) as f:
        names = f.readline().lstrip("#").split()
    ch = [np.loadtxt(fn) for fn in files]
    n = min(len(c) for c in ch)
    ch = np.stack([c[:n] for c in ch])
    pn = [p for p in names if p != "log_posterior"]
    return pn, ch[:, :, : len(pn)].reshape(-1, len(pn))


def sigma_from_p(p):
    return float(stats.norm.isf(max(p, 1e-300) / 2.0))


zs = {}
zvecs = []
znames = None
for r in runs:
    pn, flat = load(r["prefix"])
    tr = r["truth"]
    use = [p for p in pn if p in tr]
    idx = [pn.index(p) for p in use]
    x = flat[:, idx]
    t = np.array([tr[p] for p in use])
    mean, std = x.mean(0), x.std(0)
    z = (mean - t) / std
    zvecs.append(z)
    znames = use

    print(f"\n=== {r.get('label', r['prefix'])}  ({len(flat)} samples) ===")
    print(f"{'param':8s} {'truth':>10s} {'mean':>10s} {'std':>10s} {'z':>8s} {'q_sigma':>8s}")
    for k, p in enumerate(use):
        q = np.mean(x[:, k] < t[k])
        p2 = 2 * min(q, 1 - q)
        print(f"{p:8s} {t[k]:10.5g} {mean[k]:10.5g} {std[k]:10.3g} {z[k]:+8.2f} {sigma_from_p(p2):8.2f}")
        zs.setdefault(p, []).append(z[k])

    cov = np.cov(x.T)
    d = mean - t
    d2 = float(d @ np.linalg.solve(cov, d))
    pval = stats.chi2.sf(d2, df=len(use))
    print(f"Joint ({len(use)} params): Mahalanobis^2 = {d2:.2f}, p = {pval:.3f}, "
          f"equivalent = {sigma_from_p(pval):.2f} sigma")

if len(runs) >= 3:
    n = len(runs)
    print(f"\n=== Aggregate over {n} runs (expect mean 0 +/- {1/np.sqrt(n):.2f}, std ~1) ===")
    print(f"{'param':8s} {'mean z':>8s} {'std z':>8s}")
    for p, v in zs.items():
        v = np.array(v)
        print(f"{p:8s} {v.mean():+8.2f} {v.std(ddof=1):8.2f}")

# Hotelling T^2 on the per-run z vectors: tests whether the mean offset vector is zero,
# using the run-to-run scatter (no Gaussian-product assumption, correlations included).
Zm = np.array(zvecs)
Nr, K = Zm.shape
if Nr > K + 1:
    def hotelling(Zs, label):
        n, k = Zs.shape
        m = Zs.mean(0)
        S = np.cov(Zs.T).reshape(k, k)
        t2 = float(n * m @ np.linalg.solve(S, m))
        F = (n - k) / (k * (n - 1)) * t2
        print(f"{label}: T^2 = {t2:.2f}, F({k},{n-k}) p = {stats.f.sf(F, k, n - k):.3f}")
    print("\n=== Hotelling T^2 on per-run z vectors (N=%d) ===" % Nr)
    se = Zm.std(0, ddof=1) / np.sqrt(Nr)
    for p, mz, e in zip(znames, Zm.mean(0), se):
        print(f"{p:8s} mean z = {mz:+.2f} +/- {e:.2f}  ({mz / e:+.1f} sigma)")
    hotelling(Zm, f"all {K} params")
    if "As" in znames and "omch2" in znames:
        idx = [znames.index("As"), znames.index("omch2")]
        hotelling(Zm[:, idx], "As-omch2 pair")