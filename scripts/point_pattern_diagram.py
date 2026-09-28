"""Plain-language figure for the point-pattern tests (notebook 04, section 5).

Row 1: three made-up dot patterns (clustered, random, regular) with nearest-neighbour lines and Clark-Evans R.
Row 2: the real rock turned into dots: grain outlines with centre dots and a counting circle (zoom);
       the whole image with the largest 20 % of pores in red.

Run from the project root:  python scripts/point_pattern_diagram.py
Output: outputs/figures/point_patterns_simple.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from skimage import segmentation as sg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from heterogeneity import clark_evans  # noqa: E402
from segmentation import load_rgb  # noqa: E402

IMAGE = os.path.join(ROOT, "Lamina_alunos_grande.png")
LABELS = os.path.join(ROOT, "outputs", "labels_grain_large.npy")
OBJS = os.path.join(ROOT, "outputs", "03_objects_scaled.csv")
OUT = os.path.join(ROOT, "outputs", "figures", "point_patterns_simple.png")
S, NPTS = 400, 120
COL = {"CLUSTERED": "firebrick", "RANDOM": "0.3", "REGULAR": "tab:green"}


def patterns(rng):
    rand = rng.uniform(0, S, (NPTS, 2))
    centres = rng.uniform(60, S - 60, (8, 2))
    clus = np.clip(centres[rng.integers(0, 8, NPTS)] + rng.normal(0, 14, (NPTS, 2)), 0, S)
    g = np.arange(S / 11 / 2, S, S / 11)
    reg = np.array([(x, y) for x in g for y in g])[:NPTS] + rng.normal(0, 3, (NPTS, 2))
    return {"CLUSTERED": clus, "RANDOM": rand, "REGULAR": np.clip(reg, 0, S)}


def main():
    rng = np.random.default_rng(2)
    pats = patterns(rng)
    fig = plt.figure(figsize=(19, 13))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.3, wspace=0.18)

    # ---- row 1: three made-up patterns
    for k, (name, pts) in enumerate(pats.items()):
        a = fig.add_subplot(gs[0, k])
        d, j = cKDTree(pts).query(pts, k=2)
        for p, q in zip(pts, pts[j[:, 1]]):
            a.plot([p[0], q[0]], [p[1], q[1]], color="0.7", lw=1)
        a.scatter(pts[:, 0], pts[:, 1], s=22, color=COL[name], zorder=3)
        c = pts[np.argmin(np.hypot(*(pts - S / 2).T))]
        a.add_patch(plt.Circle(c, 40, fill=False, ec="blue", lw=2, ls="--"))
        n_in = int((np.hypot(*(pts - c).T) <= 40).sum() - 1)
        a.text(c[0] + 42, c[1] + 42, f"{n_in} neighbours\nwithin r = 40", color="blue", fontsize=10,
               bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        R = clark_evans(pts, S, S)["R"]
        verdict = {"CLUSTERED": "R < 1: neighbours CLOSER than random",
                   "RANDOM": "R ~ 1: like random", "REGULAR": "R > 1: neighbours FURTHER than random"}[name]
        a.set_title(f"({'abc'[k]}) {name} dots (made-up example)\nClark-Evans R = {R:.2f}  ->  {verdict}",
                    fontsize=12, color=COL[name])
        a.set_xlim(0, S)
        a.set_ylim(0, S)
        a.set_aspect("equal")
        a.set_xticks([])
        a.set_yticks([])
    fig.text(0.5, 0.955, "grey lines = each dot to its nearest neighbour;  blue circle = counting the neighbours within "
             "a distance r (the idea of Ripley's test)", ha="center", fontsize=12)

    # ---- row 2: the real rock as dots
    rgb = load_rgb(IMAGE)
    lab = np.load(LABELS)
    objs = pd.read_csv(OBJS)
    big = objs[(objs.image == "large") & (objs.area >= 12)]
    g = big[big.phase == "grain"]
    y0, x0, h, w = 420, 560, 150, 210
    a = fig.add_subplot(gs[1, 0])
    a.imshow(sg.mark_boundaries(rgb[y0:y0 + h, x0:x0 + w], lab[y0:y0 + h, x0:x0 + w], color=(1, 1, 0)))
    gi = g[(g.cx > x0) & (g.cx < x0 + w) & (g.cy > y0) & (g.cy < y0 + h)]
    a.scatter(gi.cx - x0, gi.cy - y0, s=30, color="red", edgecolor="k", zorder=3)
    c = gi[["cx", "cy"]].values[np.argmin(np.hypot(gi.cx - x0 - w / 2, gi.cy - y0 - h / 2))] - [x0, y0]
    a.add_patch(plt.Circle(c, 35, fill=False, ec="blue", lw=2.5, ls="--"))
    a.set_title("(d) the real rock (zoom): each grain -> ONE dot at its centre\n(yellow = grain outline, red = centre; "
                "blue = counting circle)", fontsize=12)
    a.axis("off")

    p = big[big.phase == "pore"]
    large = p[p.area_u >= p.area_u.quantile(0.8)]
    a = fig.add_subplot(gs[1, 1:3])
    a.imshow(rgb * 0.45 + 0.55)
    a.scatter(p.cx, p.cy, s=6, color="0.35", label=f"all pore bodies ({len(p)})")
    a.scatter(large.cx, large.cy, s=26, color="red", edgecolor="k", lw=0.4, label=f"largest 20 % of pores ({len(large)})")
    a.legend(loc="upper right", fontsize=11, framealpha=0.95)
    a.set_title("(e) the whole image as dots. Question of the right-hand notebook plot: are the RED dots (largest "
                "pores)\ngrouped together more than a random choice of grey dots would be?", fontsize=12)
    a.axis("off")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=78, bbox_inches="tight")
    print("saved", OUT)


if __name__ == "__main__":
    main()
