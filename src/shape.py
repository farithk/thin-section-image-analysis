"""Fourier shape analysis of grain outlines.

1. Centroid-radius signature r(theta) (Ehrlich & Weinberg, 1970, J. Sed. Petrol. 40, 205-212):
   radii from the area centroid to the outline at N equally spaced angles, normalised by the
   mean radius, then decomposed into harmonics. Harmonic n has amplitude A_n (fraction of the
   mean radius):
     n = 1        centroid offset (~0 by construction, a quality check)
     n = 2-4      gross form: elongation (2), triangularity (3), squareness (4)
     n = 5-12     roundness / angularity of corners
     n > 12       surface texture (limited by image resolution)
   Only defined for star-shaped outlines, where each ray from the centroid meets the outline once.
2. Elliptic Fourier descriptors (Kuhl & Giardina, 1982, Comput. Graph. Image Process. 18, 236-258):
   the x(t), y(t) outline coordinates are decomposed along the arc length. Valid for any closed outline,
   and normalised to be invariant to size, rotation and starting point.
"""
import numpy as np
import pyefd
from scipy import ndimage as ndi
from skimage import measure


def grain_contours(labels, ids=None, pad=2):
    """Sub-pixel outline (N x 2, as x, y in image coordinates) for each label id."""
    out = {}
    for lab, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None or (ids is not None and lab not in ids):
            continue
        m = np.pad(labels[sl] == lab, pad).astype(float)
        cs = measure.find_contours(m, 0.5)
        c = max(cs, key=len)                               # outer boundary
        if np.allclose(c[0], c[-1]):
            c = c[:-1]                                     # find_contours repeats the first point
        xy = np.column_stack([c[:, 1] + sl[1].start - pad, c[:, 0] + sl[0].start - pad])
        out[lab] = xy
    return out


def polygon_centroid(xy):
    x, y = xy[:, 0], xy[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    A = cr.sum() / 2
    return np.array([((x + x1) * cr).sum() / (6 * A), ((y + y1) * cr).sum() / (6 * A)])


def angular_backtrack(xy):
    """Largest backward step (deg) of the polar angle along the outline, seen from the centroid.
    0 for a strictly star-shaped outline; a few degrees = pixel noise or a tiny concavity."""
    d = xy - polygon_centroid(xy)
    th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    if th[-1] < th[0]:
        th = -th
    return float(np.degrees(np.max(np.maximum.accumulate(th) - th)))


def radial_signature(xy, n_angles=64, max_backtrack_deg=5.0):
    """r(theta) at n_angles equally spaced angles from the area centroid.
    Returns (r, is_star). Outlines whose polar angle backtracks by more than max_backtrack_deg
    are not star-shaped; for them r is the outermost radius per angle (outer envelope)."""
    c = polygon_centroid(xy)
    d = xy - c
    theta = np.arctan2(d[:, 1], d[:, 0])
    r = np.hypot(d[:, 0], d[:, 1])
    is_star = angular_backtrack(xy) <= max_backtrack_deg
    grid = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
    if is_star:
        o = np.argsort(theta)
        rr = np.interp(grid, theta[o], r[o], period=2 * np.pi)
    else:
        # densify the outline, then take the outermost radius in each angular bin
        t = np.linspace(0, 1, 20 * len(xy))
        s = np.linspace(0, 1, len(xy) + 1)
        closed = np.vstack([d, d[:1]])
        dd = np.column_stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])])
        th, rd = np.arctan2(dd[:, 1], dd[:, 0]), np.hypot(dd[:, 0], dd[:, 1])
        b = ((th + np.pi) / (2 * np.pi) * n_angles).astype(int) % n_angles
        rr = np.zeros(n_angles)
        np.maximum.at(rr, b, rd)
    return rr, is_star


def harmonic_amplitudes(r, n_max=24):
    """Amplitudes A_1..A_n_max of r(theta)/mean(r) (fraction of the mean radius)."""
    R = r / r.mean()
    F = np.fft.rfft(R) / len(R)
    return 2 * np.abs(F[1:n_max + 1])


def efd(xy, order=20):
    """Normalised elliptic Fourier coefficients (order x 4)."""
    return pyefd.elliptic_fourier_descriptors(np.vstack([xy, xy[:1]]), order=order, normalize=True)


def efd_power(coeffs):
    """Harmonic power of normalised EFD coefficients: (a^2 + b^2 + c^2 + d^2) / 2."""
    return (coeffs ** 2).sum(axis=1) / 2


def reconstruct(coeffs, n_points=200):
    return pyefd.reconstruct_contour(coeffs, locus=(0, 0), num_points=n_points)
