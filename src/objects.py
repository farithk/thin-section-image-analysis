"""Split binary phase masks (grains or pores) into individual objects.

Marker-controlled watershed on the Euclidean distance transform (EDT):
    1. EDT of the mask: each pixel's distance to the nearest boundary.
    2. Light Gaussian smoothing of the EDT (removes spurious peaks from rough outlines).
    3. Markers = regional maxima with dynamic >= h (h-maxima transform). A neck between
       two bodies is kept as a split only if both bodies are at least h px "thicker" than the neck.
    4. Watershed of -EDT from the markers, restricted to the mask.
    5. Neck-ratio merge: two adjacent regions are merged back when the neck between them
       (max EDT on their shared boundary = saddle) is wide relative to the smaller region's
       inscribed radius (saddle / min(peak_i, peak_j) >= neck_ratio). A true grain-grain
       contact is a narrow neck; a waist inside one elongated grain is a wide one.
       The absolute h alone cannot tell these apart because it does not scale with object size.

For pores this is the idea behind the SNOW algorithm (Gostick, 2017, Phys. Rev. E 96, 023307):
the pore space is partitioned into pore bodies whose watershed lines are the pore throats.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation


def split_objects(mask, sigma=1.0, h=0.5, neck_ratio=0.7, min_size=10, return_steps=False):
    """Return an int32 label image of the objects in `mask`.
    With return_steps=True, also return a dict of the intermediate results (for figures)."""
    dt = ndi.distance_transform_edt(mask)
    dts = ndi.gaussian_filter(dt, sigma) if sigma > 0 else dt
    peaks = morphology.h_maxima(dts, h) & mask
    markers, _ = ndi.label(peaks)
    # Mask components too thin to contain an h-maximum still need one marker each.
    comp, n = ndi.label(mask)
    missing = np.setdiff1d(np.arange(1, n + 1), np.unique(comp[markers > 0]))
    if missing.size:
        extra = np.isin(comp, missing)
        extra_lab, _ = ndi.label(extra)
        markers[extra] = extra_lab[extra] + markers.max()
    labels = segmentation.watershed(-dts, markers, mask=mask)
    watershed_labels = labels
    if neck_ratio is not None:
        labels = merge_wide_necks(labels, dt, neck_ratio)
    labels = morphology.remove_small_objects(labels, max_size=min_size)
    labels, _, _ = segmentation.relabel_sequential(labels)
    if return_steps:
        return labels, {"components": comp, "dt": dt, "dt_smooth": dts, "markers": markers,
                        "watershed": watershed_labels}
    return labels


def neck_table(labels, dt):
    """Adjacent label pairs (i, j) with neck = max EDT on their shared boundary (the saddle)."""
    pairs = []
    for a, b, da, db in ((labels[:, :-1], labels[:, 1:], dt[:, :-1], dt[:, 1:]),
                         (labels[:-1, :], labels[1:, :], dt[:-1, :], dt[1:, :])):
        m = (a != b) & (a > 0) & (b > 0)
        pairs.append(pd.DataFrame({"i": np.minimum(a[m], b[m]), "j": np.maximum(a[m], b[m]),
                                   "neck": np.minimum(da[m], db[m])}))
    return pd.concat(pairs).groupby(["i", "j"])["neck"].max().reset_index()


def throat_table(labels, mask):
    """Throats between adjacent pore bodies. Width = 2 x inscribed radius at the saddle (px)."""
    t = neck_table(labels, ndi.distance_transform_edt(mask))
    t["width"] = 2 * t["neck"]
    return t


def merge_wide_necks(labels, dt, neck_ratio):
    """Merge adjacent regions whose shared neck is wide relative to their size (see module doc)."""
    edges = neck_table(labels, dt)
    if edges.empty:
        return labels
    peak = ndi.maximum(dt, labels, index=np.arange(labels.max() + 1))
    edges["ratio"] = edges["neck"] / np.minimum(peak[edges["i"]], peak[edges["j"]])

    parent = np.arange(labels.max() + 1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges.loc[edges["ratio"] >= neck_ratio, ["i", "j"]].itertuples(index=False):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(x) for x in range(parent.size)])
    return roots[labels]


def touches_border(labels):
    """Set of labels touching the image border."""
    edge = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    return set(np.unique(edge[edge > 0]).tolist())


def object_table(labels, phase_name=""):
    """Per-object morphometry (pixels). Border-touching objects are flagged, not removed."""
    props = measure.regionprops_table(
        labels,
        properties=("label", "area", "perimeter", "perimeter_crofton", "equivalent_diameter_area",
                    "major_axis_length", "minor_axis_length", "orientation",
                    "solidity", "eccentricity", "centroid"),
    )
    df = pd.DataFrame(props).rename(columns={"equivalent_diameter_area": "eq_diameter",
                                             "centroid-0": "cy", "centroid-1": "cx"})
    df["aspect_ratio"] = df["major_axis_length"] / df["minor_axis_length"].replace(0, np.nan)
    # Crofton perimeter: the default pixel-count perimeter underestimates on small rounded
    # objects (circularity up to ~1.2 for a digital disc of r=3); Crofton stays ~1.0.
    df["circularity"] = 4 * np.pi * df["area"] / df["perimeter_crofton"].replace(0, np.nan) ** 2
    df["orientation_deg"] = np.degrees(df["orientation"])
    df["on_border"] = df["label"].isin(touches_border(labels))
    df["phase"] = phase_name
    return df
