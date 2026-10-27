"""
Build an EXTERNAL 240-point theory vector (pyccl + CAMB) in the same layout as
make_dv.py / the gholax mock data vector, and optionally compare it to gholax's own.

  C_ell      : pyccl (Limber, halofit by default), shear spectra
  transform  : CAMB's exact correlation routine (camb.correlations.cl2corr),
               Gauss-averaged uniformly in cos(theta) over each bin (same bin edges as gholax)
  Nothing from gholax is used except the bin layout and n(z) shapes.

Layout: w_theta (lens 0,1 autos), gamma_t (lens i x source j, i outer), xi_plus and xi_minus
(source pairs 00, 01, 11); 20 theta bins each.

Usage:
  python make_pyccl_theory.py --out theory_pyccl.npy --compare theory_fid.npy
  options: --smag 1.0 (magnification slope s; 0.4 = no magnification), --pk halofit|linear,
           --b1 1.5 1.7, --mnu 0.01, --frac 0.10 (error used for the chi^2 estimate)
Needs: pyccl, camb, numpy.
"""
import argparse
import numpy as np
import pyccl as ccl
from camb import correlations

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="theory_pyccl.npy")
ap.add_argument("--compare", default=None)
ap.add_argument("--smag", type=float, default=1.0)
ap.add_argument("--b1", type=float, nargs=2, default=[1.5, 1.7])
ap.add_argument("--pk", default="halofit", choices=["halofit", "linear"])
ap.add_argument("--mnu", type=float, default=0.01)
ap.add_argument("--no-s8z-bias", action="store_true",
                help="use constant b1 instead of b1/D(z). Default mimics gholax scale_by_s8z: true")
ap.add_argument("--frac", type=float, default=0.10)
args = ap.parse_args()

L_MAX, N_THETA, N_GL = 3001, 20, 64
_trapz = getattr(np, "trapezoid", None) or np.trapz

# --- cosmology (values from the gholax config) -------------------------------------
H0, ombh2, omch2, As, ns = 67.7, 0.022447, 0.11923, 2.107e-9, 0.9649
kw = dict(h=H0 / 100, Omega_b=ombh2 / (H0 / 100) ** 2, Omega_c=omch2 / (H0 / 100) ** 2, A_s=As, n_s=ns,
          m_nu=args.mnu, mass_split="equal", matter_power_spectrum=args.pk)
cosmo = None
for tf in ["boltzmann_class", "boltzmann_camb"]:
    try:
        cosmo = ccl.Cosmology(transfer_function=tf, **kw)
        ccl.sigma8(cosmo)
        print("transfer function:", tf)
        break
    except Exception as e:
        print(f"{tf} failed ({type(e).__name__})")
        cosmo = None
assert cosmo is not None, "no working Boltzmann code for pyccl"

# --- n(z) exactly as in make_dv.py ----------------------------------------------------
z = np.linspace(0.0, 3.0, 200)
def gauss(zm, s):
    n = np.exp(-0.5 * ((z - zm) / s) ** 2)
    return n / _trapz(n, z)
# gholax config has RealSpaceBiasExpansion.scale_by_s8z: true -> the effective bias is b1 / D(z)
# (b1*sigma8(z) held fixed), which is why a constant pyccl bias of ~b1 does not match.
Dz = ccl.growth_factor(cosmo, 1.0 / (1.0 + z))
def bias_z(b):
    return b * np.ones_like(z) if args.no_s8z_bias else b / Dz
print("effective bias at lens means:", [round(float(np.interp(m, z, bias_z(b))), 3) for m, b in zip([0.3, 0.5], args.b1)])
lens = [ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z, gauss(m, s)),
                               bias=(z, bias_z(b)),
                               mag_bias=(z, args.smag * np.ones_like(z)))
        for m, s, b in zip([0.3, 0.5], [0.05, 0.05], args.b1)]
src = [ccl.WeakLensingTracer(cosmo, dndz=(z, gauss(m, s))) for m, s in zip([0.7, 1.0], [0.1, 0.1])]

# --- C_ell on a grid, interpolated to every integer ell ---------------------------------
ellg = np.unique(np.concatenate([np.arange(2, 41), np.round(np.logspace(np.log10(41), np.log10(L_MAX), 140))])).astype(float)
ell = np.arange(2, L_MAX + 1)
def cl(t1, t2):
    c = ccl.angular_cl(cosmo, t1, t2, ellg)
    if np.all(c > 0):
        return np.exp(np.interp(np.log(ell), np.log(ellg), np.log(c)))
    return np.interp(np.log(ell), np.log(ellg), c)

# --- theta bins and Gauss nodes (same edge construction as gholax) -------------------------
centers = np.deg2rad(np.logspace(np.log10(10 / 60), np.log10(300 / 60), N_THETA))
half = np.diff(centers) / 2
edges = np.concatenate([[centers[0] - half[0]], centers[:-1] + half, [centers[-1] + half[-1]]])
mu = np.clip(np.cos(edges), -1 + 1e-6, 1 - 1e-6)
xg, wg = np.polynomial.legendre.leggauss(N_GL)
nodes = np.zeros((N_THETA, N_GL))
for b in range(N_THETA):
    lo, hi = mu[b + 1], mu[b]
    nodes[b] = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
fac = ell * (ell + 1) / (2 * np.pi)

def to_cf(c, col):
    cls = np.zeros((L_MAX + 1, 4))
    cls[2:, col] = c * fac
    corr = correlations.cl2corr(cls, nodes.ravel(), lmax=L_MAX)
    return (corr[:, {0: 0, 1: 1, 3: 3}[col]].reshape(N_THETA, N_GL) * (wg / 2.0)).sum(1)

def to_cf_minus(c):
    cls = np.zeros((L_MAX + 1, 4)); cls[2:, 1] = c * fac
    corr = correlations.cl2corr(cls, nodes.ravel(), lmax=L_MAX)
    return (corr[:, 2].reshape(N_THETA, N_GL) * (wg / 2.0)).sum(1)

w = [to_cf(cl(lens[i], lens[i]), 0) for i in range(2)]
gam = [to_cf(cl(lens[i], src[j]), 3) for i in range(2) for j in range(2)]
pairs = [(0, 0), (0, 1), (1, 1)]
xp = [to_cf(cl(src[i], src[j]), 1) for i, j in pairs]
xm = [to_cf_minus(cl(src[i], src[j])) for i, j in pairs]
vec = np.concatenate(w + gam + xp + xm)
assert vec.shape == (240,)
np.save(args.out, vec)
print("saved", args.out, vec.shape)

# --- optional comparison with gholax's vector -------------------------------------------------
if args.compare:
    g = np.load(args.compare)
    names = ["w0", "w1", "g00", "g01", "g10", "g11", "xi+00", "xi+01", "xi+11", "xi-00", "xi-01", "xi-11"]
    cuts = {"w": (0.008, 0.083), "g": (0.007, 0.085), "x": (0.008, 0.078)}
    tot, npts = 0.0, 0
    print(f"\nratio external/gholax, bins 0..19 (every 2nd), and chi2 with {args.frac:.0%} errors inside the config scale cuts")
    for k, nm in enumerate(names):
        a, b = vec[20 * k:20 * (k + 1)], g[20 * k:20 * (k + 1)]
        lo, hi = cuts[nm[0] if nm[0] in "wg" else "x"]
        m = (centers > lo) & (centers < hi)
        chi2 = np.sum(((a - b) / (args.frac * np.abs(b)))[m] ** 2)
        tot += chi2; npts += m.sum()
        print(f"{nm:6s} " + " ".join(f"{x:5.2f}" for x in (a / b)[::2]) + f"   chi2={chi2:8.1f} ({m.sum()} pts)")
    print(f"TOTAL chi2 = {tot:.1f} for {npts} points  (0 would mean identical predictions)")