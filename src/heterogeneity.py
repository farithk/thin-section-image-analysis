"""Spatial heterogeneity tools for binary phase maps and object centroids.

- REV by random windows (Bear, 1972): porosity statistics against window size.
- Two-point correlation S2(r) of a phase indicator (Torquato, 2002). The indicator
  semivariogram is gamma(r) = phi - S2(r).
- Gliding-box lacunarity (Allain & Cloitre, 1991; Plotnick et al., 1996).
- Point patterns: Clark & Evans (1954) nearest-neighbour index with the Donnelly (1978)
  edge correction, and Ripley's K / Besag's L with border correction and Monte Carlo envelopes.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ---------------------------------------------------------------- windows / REV
def integral_image(a):
    ii = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
    ii[1:, 1:] = a.astype(float).cumsum(0).cumsum(1)
    return ii


def box_sums(ii, s):
    """Sums of all s x s gliding boxes (valid positions only)."""
    return ii[s:, s:] - ii[:-s, s:] - ii[s:, :-s] + ii[:-s, :-s]


def rev_random_windows(mask, sizes, n=400, seed=0):
    """Mean / std / CV of the phase fraction in n random square windows per size."""
    rng = np.random.default_rng(seed)
    ii = integral_image(mask)
    H, W = mask.shape
    rows = []
    for s in sizes:
        if s > min(H, W):
            continue
        y = rng.integers(0, H - s + 1, n)
        x = rng.integers(0, W - s + 1, n)
        v = (ii[y + s, x + s] - ii[y, x + s] - ii[y + s, x] + ii[y, x]) / s ** 2
        rows.append({"L": s, "mean": v.mean(), "std": v.std(ddof=1),
                     "p5": np.percentile(v, 5), "p95": np.percentile(v, 95)})
    df = pd.DataFrame(rows)
    df["cv"] = df["std"] / df["mean"]
    return df


def moving_window_fraction(mask, s):
    """Phase fraction in a centred s x s window at every pixel (edges: reflected)."""
    from scipy.ndimage import uniform_filter
    return uniform_filter(mask.astype(float), size=s, mode="reflect")


# ---------------------------------------------------------------- two-point statistics
def two_point_map(mask):
    """S2(dy, dx) by FFT with zero padding, normalised by the number of overlapping pairs.
    Returns (S2 map centred at zero lag, phi)."""
    f = mask.astype(float)
    H, W = f.shape
    shape = (2 * H, 2 * W)
    F = np.fft.rfft2(f, shape)
    O = np.fft.rfft2(np.ones_like(f), shape)
    num = np.fft.irfft2(F * np.conj(F), shape)
    den = np.fft.irfft2(O * np.conj(O), shape)
    s2 = np.fft.fftshift(num / np.maximum(den, 1))
    return s2, f.mean()


def radial_profile(m, rmax):
    cy, cx = np.array(m.shape) // 2
    y, x = np.indices(m.shape)
    r = np.hypot(y - cy, x - cx).round().astype(int)
    sel = r <= rmax
    return np.bincount(r[sel], m[sel]) / np.bincount(r[sel])


def correlation_functions(mask, rmax=200):
    """Radial and directional (x = horizontal, y = vertical) normalised autocorrelation
    C(r) = (S2 - phi^2) / (phi - phi^2): 1 at r=0, 0 when uncorrelated."""
    s2, phi = two_point_map(mask)
    cy, cx = np.array(s2.shape) // 2
    norm = lambda v: (v - phi ** 2) / (phi - phi ** 2)
    r = np.arange(rmax + 1)
    return pd.DataFrame({"r": r,
                         "C_radial": norm(radial_profile(s2, rmax)),
                         "C_x": norm(s2[cy, cx:cx + rmax + 1]),
                         "C_y": norm(s2[cy:cy + rmax + 1, cx]),
                         "gamma_radial": phi - radial_profile(s2, rmax)}), phi


def correlation_length(r, c, level=np.exp(-1)):
    """First lag where C(r) drops below `level` (linear interpolation)."""
    i = np.argmax(c < level)
    if i == 0:
        return np.nan
    return r[i - 1] + (c[i - 1] - level) / (c[i - 1] - c[i]) * (r[i] - r[i - 1])


# ---------------------------------------------------------------- lacunarity
def lacunarity(mask, sizes):
    """Gliding-box lacunarity Lambda(r) = E[M^2] / E[M]^2 of the box mass M."""
    ii = integral_image(mask)
    out = []
    for s in sizes:
        m = box_sums(ii, s).ravel()
        out.append(np.mean(m ** 2) / np.mean(m) ** 2)
    return np.array(out)


# ---------------------------------------------------------------- point patterns
def clark_evans(pts, width, height):
    """Clark-Evans R with Donnelly edge correction. R < 1 clustered, R > 1 regular."""
    n = len(pts)
    A, P = width * height, 2 * (width + height)
    d, _ = cKDTree(pts).query(pts, k=2)
    obs = d[:, 1].mean()
    exp = 0.5 * np.sqrt(A / n) + (0.0514 + 0.041 / np.sqrt(n)) * P / n
    se = np.sqrt(0.0703 * A / n ** 2 + 0.037 * P * np.sqrt(A / n ** 5))
    z = (obs - exp) / se
    from scipy.stats import norm
    return {"n": n, "mean_nn": obs, "expected_csr": exp, "R": obs / exp, "z": z,
            "p_two_sided": 2 * norm.sf(abs(z))}


def ripley_L(pts, width, height, radii):
    """Besag L(r) - r with border (reduced-sample) edge correction."""
    pts = np.asarray(pts, float)
    n, A = len(pts), width * height
    lam = n / A
    tree = cKDTree(pts)
    bdist = np.minimum.reduce([pts[:, 0], width - pts[:, 0], pts[:, 1], height - pts[:, 1]])
    out = []
    for r in radii:
        inner = bdist >= r
        if inner.sum() < 2:
            out.append(np.nan)
            continue
        cnt = np.array([len(x) - 1 for x in tree.query_ball_point(pts[inner], r)])
        K = cnt.mean() / lam
        out.append(np.sqrt(K / np.pi) - r)
    return np.array(out)


def ripley_envelope(n, width, height, radii, nsim=99, seed=0, pool=None):
    """Pointwise min/max envelope of L(r)-r.
    pool=None: complete spatial randomness (CSR) with n points.
    pool=array: random labelling, i.e. n points drawn without replacement from `pool`."""
    rng = np.random.default_rng(seed)
    sims = []
    for _ in range(nsim):
        if pool is None:
            p = np.column_stack([rng.uniform(0, width, n), rng.uniform(0, height, n)])
        else:
            p = pool[rng.choice(len(pool), n, replace=False)]
        sims.append(ripley_L(p, width, height, radii))
    sims = np.array(sims)
    return np.nanmin(sims, 0), np.nanmax(sims, 0), np.nanmean(sims, 0)
