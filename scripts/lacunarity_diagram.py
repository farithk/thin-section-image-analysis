"""Plain-language figure for lacunarity (notebook 04, section 4).

Row 1: two made-up images with the same porosity (19 %): pores spread evenly vs pores gathered in zones,
       with a grid of boxes coloured by the porosity inside each box.
Row 2: histograms of the box porosities and the lacunarity of each image.
Row 3: the real pore map, the same pore bodies moved to random places, and their lacunarity curves.

Run from the project root:  python scripts/lacunarity_diagram.py
Output: outputs/figures/lacunarity_simple.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from heterogeneity import lacunarity  # noqa: E402

LABELS = os.path.join(ROOT, "outputs", "labels_pore_large.npy")
OUT = os.path.join(ROOT, "outputs", "figures", "lacunarity_simple.png")
N, R, BOX = 400, 5, 50


def disks(centers, n=N, r=R):
    yy, xx = np.mgrid[:n, :n]
    img = np.zeros((n, n), bool)
    for cy, cx in centers:
        img |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    return img


def fill_to(target, sampler, rng):
    """Add disks one at a time until the porosity reaches the target."""
    img = np.zeros((N, N), bool)
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    disk = yy ** 2 + xx ** 2 <= R * R
    while img.mean() < target:
        cy, cx = sampler(rng)
        y0, x0 = cy - R, cx - R
        ys, xs = slice(max(y0, 0), min(y0 + 2 * R + 1, N)), slice(max(x0, 0), min(x0 + 2 * R + 1, N))
        img[ys, xs] |= disk[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
    return img


def box_grid(img, b=BOX):
    k = img.shape[0] // b
    return img[:k * b, :k * b].reshape(k, b, k, b).mean(axis=(1, 3))


def lac_of_boxes(v):
    return np.mean(v ** 2) / np.mean(v) ** 2


def relocate(L, rng, tries=200):
    out = np.zeros(L.shape, bool)
    objs = [(L[sl] == lab) for sl, lab in zip(ndi.find_objects(L), range(1, L.max() + 1)) if sl is not None]
    for patch in sorted(objs, key=lambda q: -q.sum()):
        h, w = patch.shape
        for _ in range(tries):
            y, x = rng.integers(0, L.shape[0] - h + 1), rng.integers(0, L.shape[1] - w + 1)
            if not (out[y:y + h, x:x + w] & patch).any():
                out[y:y + h, x:x + w] |= patch
                break
    return out


def main():
    rng = np.random.default_rng(4)
    even = fill_to(0.19, lambda g: g.integers(0, N, 2), rng)
    zones = [(90, 110), (300, 290), (120, 320), (310, 90)]
    clustered = fill_to(0.19, lambda g: np.clip(np.array(zones[g.integers(0, 4)]) + g.normal(0, 38, 2), 0, N - 1).astype(int), rng)

    fig = plt.figure(figsize=(19, 17))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 0.8, 1], hspace=0.35, wspace=0.25)

    # ---- rows 1-2: the made-up example
    for k, (img, name) in enumerate([(even, "EVENLY SPREAD pores"), (clustered, "pores GATHERED IN ZONES")]):
        g = box_grid(img)
        a = fig.add_subplot(gs[0, 2 * k])
        a.imshow(~img, cmap="gray", vmin=0, vmax=1)
        for t in range(0, N + 1, BOX):
            a.axhline(t - 0.5, color="red", lw=0.6)
            a.axvline(t - 0.5, color="red", lw=0.6)
        a.set_title(f"({'ab'[k]}) {name}\nporosity {100 * img.mean():.0f} %  (black = pore); red grid = boxes",
                    fontsize=12)
        a.axis("off")
        a = fig.add_subplot(gs[0, 2 * k + 1])
        im = a.imshow(100 * g, cmap="Blues", vmin=0, vmax=60)
        for (i, j), v in np.ndenumerate(g):
            a.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=8, color="w" if v > 0.35 else "k")
        a.set_title("porosity inside each box (%)", fontsize=12)
        a.axis("off")
        a = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
        a.hist(100 * g.ravel(), bins=np.arange(0, 70, 3), color="tab:blue", edgecolor="k")
        lac = lac_of_boxes(g.ravel())
        a.set_xlabel("porosity inside a box (%)")
        a.set_ylabel("number of boxes")
        a.set_xlim(0, 70)
        a.set_title(f"({'cd'[k]}) {'boxes are similar (all ~ 10-30 %)' if k == 0 else 'many almost empty boxes, a few very full ones'}"
                    f"\nLACUNARITY = {lac:.2f}  {'(close to 1: even)' if k == 0 else '(larger: gappy / clustered)'}",
                    fontsize=12, color="green" if k == 0 else "firebrick")

    # ---- row 3: the real rock
    Lp = np.load(LABELS)
    real = Lp > 0
    moved = relocate(Lp, np.random.default_rng(0))
    a = fig.add_subplot(gs[2, 0])
    a.imshow(~real, cmap="gray")
    a.set_title(f"(e) REAL pores (large image)\nporosity {100 * real.mean():.1f} %", fontsize=12)
    a.axis("off")
    a = fig.add_subplot(gs[2, 1])
    a.imshow(~moved, cmap="gray")
    a.set_title(f"(f) the SAME pores, moved at random\nporosity {100 * moved.mean():.1f} %, no zones",
                fontsize=12)
    a.axis("off")
    boxes = np.unique(np.round(np.logspace(0, np.log10(512), 18)).astype(int))
    lr, lm = lacunarity(real, boxes), lacunarity(moved, boxes)
    a = fig.add_subplot(gs[2, 2:4])
    a.plot(boxes, lr - 1, "o-", color="tab:blue", lw=2, label="real pores (e)")
    a.plot(boxes, lm - 1, "s--", color="tab:orange", lw=2, label="same pores moved at random (f)")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("box size (px)")
    a.set_ylabel("unevenness = lacunarity - 1  (0 = perfectly even)")
    a.legend(fontsize=11, loc="upper right")
    k = int(np.argmin(np.abs(boxes - 170)))
    a.annotate(f"box {boxes[k]} px: real is {(lr[k] - 1) / (lm[k] - 1):.1f}x more uneven\nthan the moved pores "
               f"-> real pores gather in porous zones",
               xy=(boxes[k], lr[k] - 1), xytext=(1.2, 0.012), fontsize=11, color="firebrick",
               arrowprops=dict(arrowstyle="->", color="firebrick"))
    a.set_title("(g) unevenness for every box size. Small boxes are always uneven (one box = one pore or one grain).\n"
                "For large boxes the REAL pores (blue) stay MORE uneven than the moved ones (orange): porous zones",
                fontsize=11, loc="left")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=80, bbox_inches="tight")
    print("saved", OUT, f"| lacunarity even {lac_of_boxes(box_grid(even).ravel()):.2f}, clustered {lac_of_boxes(box_grid(clustered).ravel()):.2f}")


if __name__ == "__main__":
    main()
