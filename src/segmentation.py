"""Three-phase segmentation of blue-epoxy impregnated thin sections.

Phases
------
0 = cement / matrix / opaques (dark, neutral colour)
1 = grain (bright, low chroma: quartz / feldspar)
2 = pore (blue epoxy: strongly negative b* in CIE Lab)

Grayscale alone cannot separate pore from cement (both are dark), so we use
CIE Lab: L* separates grains, b* separates the blue epoxy from the dark cement.
"""
from dataclasses import dataclass, field

import numpy as np
from skimage import color, filters, io, morphology

CEMENT, GRAIN, PORE = 0, 1, 2
PHASE_NAMES = {CEMENT: "cement", GRAIN: "grain", PORE: "pore"}
PHASE_COLORS = np.array([[90, 50, 30], [235, 230, 215], [30, 80, 200]], dtype=np.uint8)


def load_rgb(path):
    """Read an image and return it as float RGB in [0, 1] (alpha dropped)."""
    img = io.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[..., :3]
    return img.astype(float) / 255.0 if img.dtype == np.uint8 else img.astype(float)


def to_lab(rgb, sigma=1.0):
    """Convert to CIE Lab after light Gaussian smoothing (reduces JPEG/sensor noise)."""
    if sigma > 0:
        rgb = filters.gaussian(rgb, sigma=sigma, channel_axis=-1)
    return color.rgb2lab(rgb)


@dataclass
class ThreePhaseResult:
    phases: np.ndarray                    # int8 map with CEMENT / GRAIN / PORE
    lab: np.ndarray                       # smoothed Lab image
    t_L: float                            # L* threshold grain vs non-grain
    t_b: float                            # b* threshold pore vs cement
    fractions: dict = field(default_factory=dict)


def three_phase(rgb, sigma=1.0, t_L=None, pore_rule="hue", t_b=None,
                hue_range=(-140.0, -40.0), min_chroma=5.0, min_size=20):
    """Segment an RGB thin section into cement / grain / pore.

    Grain: L* above Otsu threshold t_L.
    Pore, pore_rule="hue" (default): Lab hue angle in the blue sector and
        chroma > min_chroma. Brightness-independent, so it also catches the
        dark navy epoxy that a plain b* threshold misses.
    Pore, pore_rule="b": b* < t_b (Otsu on the non-grain pixels by default).
    """
    lab = to_lab(rgb, sigma)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    if t_L is None:
        t_L = filters.threshold_otsu(L)
    grain = L > t_L
    if pore_rule == "hue":
        hue = np.degrees(np.arctan2(b, a))
        chroma = np.hypot(a, b)
        pore = (~grain) & (hue > hue_range[0]) & (hue < hue_range[1]) & (chroma > min_chroma)
        t_b = np.nan
    else:
        if t_b is None:
            t_b = filters.threshold_otsu(b[~grain])
        pore = (~grain) & (b < t_b)

    # Clean up: fill specks inside grains, drop tiny isolated blobs.
    grain = morphology.remove_small_holes(grain, max_size=min_size)
    grain = morphology.remove_small_objects(grain, max_size=min_size)
    pore = morphology.remove_small_objects(pore & ~grain, max_size=min_size)

    phases = np.full(L.shape, CEMENT, dtype=np.int8)
    phases[grain] = GRAIN
    phases[pore] = PORE
    return ThreePhaseResult(phases, lab, float(t_L), float(t_b), phase_fractions(phases))


def phase_fractions(phases):
    n = phases.size
    return {name: float((phases == k).sum()) / n for k, name in PHASE_NAMES.items()}


def phase_rgb(phases):
    """Colour-coded phase map for display."""
    return PHASE_COLORS[phases]
