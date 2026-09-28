"""Grain/pore size statistics.

Folk & Ward (1957) graphic measures on the phi scale, phi = -log2(D). D is in any length
unit. Sorting, skewness and kurtosis are differences/ratios of phi percentiles, so they do not
depend on the unit. The mean does (it shifts by -log2(unit)).
"""
import numpy as np

SORTING_CLASSES = [(0.35, "very well sorted"), (0.50, "well sorted"), (0.71, "moderately well sorted"),
                   (1.00, "moderately sorted"), (2.00, "poorly sorted"), (4.00, "very poorly sorted"),
                   (np.inf, "extremely poorly sorted")]


def weighted_percentile(x, q, w=None):
    """Percentiles q (0-100) of x with optional weights w."""
    x = np.asarray(x, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    order = np.argsort(x)
    x, w = x[order], w[order]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(np.asarray(q) / 100.0, cdf, x)


def folk_ward(diameters, weights=None):
    """Folk & Ward graphic measures. Percentiles are taken so that phi16 = coarse tail."""
    phi = -np.log2(np.asarray(diameters, float))
    p5, p16, p25, p50, p75, p84, p95 = weighted_percentile(phi, [5, 16, 25, 50, 75, 84, 95], weights)
    mean = (p16 + p50 + p84) / 3
    sorting = (p84 - p16) / 4 + (p95 - p5) / 6.6
    skew = ((p16 + p84 - 2 * p50) / (2 * (p84 - p16))
            + (p5 + p95 - 2 * p50) / (2 * (p95 - p5)))
    kurt = (p95 - p5) / (2.44 * (p75 - p25))
    label = next(name for lim, name in SORTING_CLASSES if sorting < lim)
    return {"mean_phi": mean, "median_phi": p50, "sorting_phi": sorting, "skewness": skew,
            "kurtosis": kurt, "sorting_class": label, "mean_D": 2 ** -mean}
