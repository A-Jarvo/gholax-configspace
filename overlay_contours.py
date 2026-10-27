"""
Overlay the posteriors of all mock runs, with the truth marked, and optionally the
Gaussian-approximation joint constraint from combining them.

Usage:  python overlay_contours.py runs.json [out.png] [--combined] [--smooth]

Thin blue lines: 68% contours of each mock (diagonal: 1D marginals).
Red lines: truth.
--combined: adds a thick black 68% ellipse = product of the per-mock Gaussian
  approximations (flat priors), i.e. the constraint from analysing all mocks together.
  This is NOT the same as pooling the samples.
"""
import glob
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
combined = "--combined" in sys.argv
smooth = "--smooth" in sys.argv   # all samples + wider KDE bandwidth -> smoother contours
BW = 0.5 if smooth else None
runs = json.load(open(argv[0]))
out = argv[1] if len(argv) > 1 else "overlay_contours.png"
rng = np.random.default_rng(0)


def load(prefix):
    files = sorted(glob.glob(f"{prefix}.samples.0.*.txt"))
    with open(files[0]) as f:
        names = f.readline().lstrip("#").split()
    ch = [np.loadtxt(fn) for fn in files]
    n = min(len(c) for c in ch)
    ch = np.stack([c[:n] for c in ch])
    pn = [p for p in names if p != "log_posterior"]
    return pn, ch[:, :, : len(pn)].reshape(-1, len(pn))


data, truths = [], []
for r in runs:
    pn, flat = load(r["prefix"])
    use = [p for p in pn if p in r["truth"]]
    data.append(flat[:, [pn.index(p) for p in use]])
    truths.append([r["truth"][p] for p in use])
names = use
truth = np.mean(truths, axis=0)     # identical across runs for a fixed fiducial
K, N = len(names), len(runs)

mu = np.array([d.mean(0) for d in data])
cov = np.array([np.cov(d.T) for d in data])
inv = np.linalg.inv(cov)
cov_c = np.linalg.inv(inv.sum(0))
mu_c = cov_c @ np.einsum("nij,nj->i", inv, mu)


def hpd_level(z, mass=0.68):
    zs = np.sort(z.ravel())[::-1]
    c = np.cumsum(zs) / zs.sum()
    return zs[np.searchsorted(c, mass)]


fig, axes = plt.subplots(K, K, figsize=(2.7 * K, 2.7 * K))
for i in range(K):
    for j in range(K):
        ax = axes[i, j]
        if j > i:
            ax.axis("off")
            continue
        for n in range(N):
            d = data[n]
            sub = d if smooth else d[rng.choice(len(d), min(1200, len(d)), replace=False)]
            if i == j:
                g = np.linspace(d[:, i].min(), d[:, i].max(), 200)
                ax.plot(g, stats.gaussian_kde(sub[:, i], bw_method=BW)(g), color="C0", alpha=0.5, lw=0.8)
            else:
                gx = np.linspace(d[:, j].min(), d[:, j].max(), 70)
                gy = np.linspace(d[:, i].min(), d[:, i].max(), 70)
                X, Y = np.meshgrid(gx, gy)
                Z = stats.gaussian_kde(sub[:, [j, i]].T, bw_method=BW)(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                ax.contour(X, Y, Z, levels=[hpd_level(Z)], colors="C0", alpha=0.5, linewidths=0.8)
        if combined:
            if i == j:
                g = np.linspace(mu_c[i] - 4 * np.sqrt(cov_c[i, i]), mu_c[i] + 4 * np.sqrt(cov_c[i, i]), 200)
                ax.plot(g, stats.norm.pdf(g, mu_c[i], np.sqrt(cov_c[i, i])), "k", lw=2)
            else:
                sel = [j, i]
                w, v = np.linalg.eigh(cov_c[np.ix_(sel, sel)])
                t = np.linspace(0, 2 * np.pi, 200)
                e = (v * np.sqrt(w * stats.chi2.ppf(0.68, 2))) @ np.vstack([np.cos(t), np.sin(t)])
                ax.plot(mu_c[j] + e[0], mu_c[i] + e[1], "k", lw=2)
        if i == j:
            ax.axvline(truth[i], color="r", lw=1)
            ax.set_yticks([])
        else:
            ax.axvline(truth[j], color="r", lw=0.8)
            ax.axhline(truth[i], color="r", lw=0.8)
        if i == K - 1:
            ax.set_xlabel(names[j])
        else:
            ax.set_xticklabels([])
        if j == 0 and i > 0:
            ax.set_ylabel(names[i])
        elif j > 0:
            ax.set_yticklabels([])

title = f"{N} mock posteriors (68%)"
if combined:
    title += "; black: joint constraint (Gaussian approx.)"
fig.suptitle(title, y=0.995)
fig.tight_layout()
fig.savefig(out, dpi=170)
print(f"Saved {out}")
if combined:
    print(f"{'param':8s} {'combined mean':>14s} {'combined std':>13s} {'single std':>11s} {'z of truth':>11s}")
    for k, p in enumerate(names):
        print(f"{p:8s} {mu_c[k]:14.5g} {np.sqrt(cov_c[k, k]):13.3g} {np.sqrt(cov[:, k, k]).mean():11.3g} "
              f"{(mu_c[k] - truth[k]) / np.sqrt(cov_c[k, k]):+11.2f}")

    dc = mu_c - truth
    d2c = float(dc @ np.linalg.solve(cov_c, dc))
    print(f"\nCombined joint: d^2 = {d2c:.2f} for {K} params, p = {stats.chi2.sf(d2c, K):.3f}")
    if "As" in names and "omch2" in names:
        sel = [names.index("As"), names.index("omch2")]
        d2 = float(dc[sel] @ np.linalg.solve(cov_c[np.ix_(sel, sel)], dc[sel]))
        rho = cov_c[sel[0], sel[1]] / np.sqrt(cov_c[sel[0], sel[0]] * cov_c[sel[1], sel[1]])
        print(f"As-omch2 pair: correlation = {rho:+.3f}, d^2 = {d2:.2f} for 2 params, p = {stats.chi2.sf(d2, 2):.3f}")