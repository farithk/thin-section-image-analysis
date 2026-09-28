"""Explanatory figure for the object split (notebook 02): watershed on the distance transform + neck-ratio merge.

For grains (row 1) and pores (row 2), on a zoomed crop of the small image:
  (1) connected components (ndi.label): touching objects are one blob
  (2) distance transform with the markers (one per candidate object)
  (3) watershed: cuts drawn at the necks
  (4) after the neck-ratio merge: red = cut kept (narrow neck), yellow = cut undone (wide neck)
A schematic on the right explains the neck-ratio rule.

Run from the project root:  python scripts/split_diagram.py
Output: outputs/figures/split_steps.png
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
from objects import split_objects  # noqa: E402
from segmentation import GRAIN, PORE  # noqa: E402

PHASES = os.path.join(ROOT, "outputs", "phases_small.npy")
OUT = os.path.join(ROOT, "outputs", "figures", "split_steps.png")
CH, CW = 80, 115
NECK_RATIO = 0.7


def cuts(labels):
    """Pixels on a boundary between two different non-zero labels."""
    b = np.zeros(labels.shape, bool)
    for a_, b_ in ((labels[1:, :], labels[:-1, :]), (labels[:, 1:], labels[:, :-1])):   # no wrap-around
        diff = (a_ > 0) & (b_ > 0) & (a_ != b_)
        if a_.shape[0] < labels.shape[0]:
            b[1:, :] |= diff
        else:
            b[:, 1:] |= diff
    return b


def random_colours(labels, seed=1, bg=(0, 0, 0)):
    rng = np.random.default_rng(seed)
    lut = rng.uniform(0.25, 1.0, (labels.max() + 1, 3))
    lut[0] = bg
    return lut[labels]


def best_crop(score):
    d = ndi.uniform_filter(score.astype(float), size=(CH, CW))
    d = d[CH // 2:-CH // 2, CW // 2:-CW // 2]
    y, x = np.unravel_index(np.argmax(d), d.shape)
    return slice(y, y + CH), slice(x, x + CW)


def main():
    ph = np.load(PHASES)
    fig = plt.figure(figsize=(22, 10.5))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.05], wspace=0.08, hspace=0.3)

    for row, (code, name) in enumerate([(GRAIN, "grains"), (PORE, "pores")]):
        mask = ph == code
        final, st = split_objects(mask, return_steps=True)
        ws_cut = cuts(st["watershed"])
        kept = ws_cut & cuts(final)
        undone = ws_cut & ~cuts(final) & mask
        sl = best_crop(kept.astype(float) + undone)                # a crop with many splits and merges
        thick = lambda m: ndi.binary_dilation(m) & mask             # wider lines, for display only
        ws_col = random_colours(st["watershed"])
        # each final object takes the colour of one of its watershed pieces -> merged pieces share a colour
        piece = ndi.maximum(st["watershed"], final, index=np.arange(final.max() + 1)).astype(int)
        final_col = ws_col.reshape(-1, 3)[0:0]
        lut = np.random.default_rng(1).uniform(0.25, 1.0, (st["watershed"].max() + 1, 3))
        lut[0] = 0
        final_col = lut[piece][final]
        n_comp, n_ws, n_final = st["components"].max(), st["watershed"].max(), final.max()

        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(random_colours(st["components"])[sl], interpolation="nearest")
        ax.set_title(f"(1) connected components\n{n_comp} {name} in the image: touching\nobjects are ONE blob",
                     fontsize=11, loc="left")

        ax = fig.add_subplot(gs[row, 1])
        dt = np.ma.masked_where(~mask, st["dt_smooth"])
        ax.imshow(dt[sl], cmap="magma", interpolation="nearest")
        my, mx = np.nonzero(st["markers"][sl] > 0)
        ax.plot(mx, my, "o", color="cyan", ms=4, mec="k", mew=0.5)
        ax.set_title("(2) distance to the edge (bright = far)\ncyan = markers: one per peak\n= one per candidate object",
                     fontsize=11, loc="left")

        ax = fig.add_subplot(gs[row, 2])
        img = ws_col[sl].copy()
        img[thick(ws_cut)[sl]] = (1, 1, 1)
        ax.imshow(img, interpolation="nearest")
        ax.set_title(f"(3) watershed: water rises from each\nmarker; cuts (white) where they meet\n{n_ws} pieces",
                     fontsize=11, loc="left")

        ax = fig.add_subplot(gs[row, 3])
        img = final_col[sl].copy()
        img[thick(undone)[sl]] = (1, 0.9, 0.0)
        img[thick(kept)[sl]] = (1, 0.1, 0.1)
        ax.imshow(img, interpolation="nearest")
        ax.set_title(f"(4) neck-ratio merge: red = cut kept,\nyellow = cut undone (merged back)\n"
                     f"final: {n_final} {name}", fontsize=11, loc="left")
        for a in fig.axes[-4:]:
            a.set_xticks([])
            a.set_yticks([])
        fig.text(0.005, 0.73 - 0.47 * row, name.upper(), fontsize=15, fontweight="bold", rotation=90, va="center")

    # schematic of the neck-ratio rule
    ax = fig.add_subplot(gs[:, 4])
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 12)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("the neck-ratio rule (step 4)", fontsize=12, loc="left", fontweight="bold")
    col = "#d9c9a3"
    # example 1: two grains touching at a narrow neck
    for cx in (2.2, 4.6):
        ax.add_patch(plt.Circle((cx, 9.3), 1.2, color=col))
    ax.add_patch(plt.Rectangle((3.3, 9.15), 0.2, 0.3, color=col))
    ax.plot([3.4, 3.4], [9.1, 9.5], color="r", lw=4)
    ax.annotate("", xy=(2.2, 8.1), xytext=(2.2, 10.5), arrowprops=dict(arrowstyle="<->", lw=1.5))
    ax.text(0.2, 10.8, "piece width W", fontsize=10)
    ax.text(3.6, 10.8, "neck n (narrow)", fontsize=10, color="r")
    ax.text(0.2, 7.0, "two grains touching:\nn / W ~ 0.15 < 0.7  ->  KEEP the cut", fontsize=11, color="r",
            fontweight="bold")
    # example 2: one elongated grain with a waist (two overlapping circles)
    for cx in (2.7, 4.1):
        ax.add_patch(plt.Circle((cx, 4.2), 1.2, color=col))
    half = np.sqrt(1.2 ** 2 - 0.7 ** 2)
    ax.plot([3.4, 3.4], [4.2 - half, 4.2 + half], color="gold", lw=4)
    ax.text(3.6, 5.7, "neck n (wide)", fontsize=10, color="#b8860b")
    ax.text(0.2, 1.9, "one grain with a 'waist':\nn / W ~ 0.8 >= 0.7  ->  UNDO the cut", fontsize=11,
            color="#b8860b", fontweight="bold")
    ax.text(0.2, 0.0, f"n = width of the contact (neck)\nW = width of the smaller piece\nthreshold n / W = {NECK_RATIO}",
            fontsize=10, family="monospace", bbox=dict(facecolor="#f4f4f4", edgecolor="0.6"))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=90, bbox_inches="tight")
    print("saved", OUT)


if __name__ == "__main__":
    main()
