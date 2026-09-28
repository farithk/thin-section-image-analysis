"""Zoomed visual check of the object split (notebook 02), one row per case:

1. touching grains are separated        -> circle on a cut that was kept (narrow neck)
2. a large elongated grain is NOT cut   -> before vs after the neck-ratio merge, circle on the undone cut
3. pores are split at their throats     -> circle on a pore throat that was cut
The circled spots are found automatically from the split itself (same split_objects() as the notebook).

Run from the project root:  python scripts/split_check_diagram.py
Output: outputs/figures/split_check.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import color

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from objects import split_objects  # noqa: E402
from segmentation import GRAIN, PORE, load_rgb  # noqa: E402

IMAGE = os.path.join(ROOT, "Lamina_alunos2.png")
PHASES = os.path.join(ROOT, "outputs", "phases_small.npy")
OUT = os.path.join(ROOT, "outputs", "figures", "split_check.png")
OUT_OVERVIEW = os.path.join(ROOT, "outputs", "figures", "merge_overview.png")
CH, CW = 90, 130


def cuts(labels):
    b = np.zeros(labels.shape, bool)
    for a_, b_ in ((labels[1:, :], labels[:-1, :]), (labels[:, 1:], labels[:, :-1])):   # no wrap-around
        diff = (a_ > 0) & (b_ > 0) & (a_ != b_)
        if a_.shape[0] < labels.shape[0]:
            b[1:, :] |= diff
        else:
            b[:, 1:] |= diff
    return b


def biggest_segment(mask, weight=None):
    """Centre (y, x) and size of the largest connected piece of `mask` (optionally weighted)."""
    lab, n = ndi.label(ndi.binary_dilation(mask, iterations=1))
    if n == 0:
        return None
    sizes = ndi.sum(mask if weight is None else weight, lab, index=np.arange(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    return ndi.center_of_mass(lab == k), sizes[k - 1]


def crop_around(yx, shape):
    y = int(np.clip(yx[0] - CH // 2, 0, shape[0] - CH))
    x = int(np.clip(yx[1] - CW // 2, 0, shape[1] - CW))
    return slice(y, y + CH), slice(x, x + CW)


def overlay(img, labels, alpha=0.45, dark=1.0):
    out = color.label2rgb(labels, img * dark, alpha=alpha, bg_label=0)
    out[cuts(labels)] = (1, 1, 1)                                   # white line between touching objects
    return out


def circle(ax, yx, sl, r=9, col="yellow"):
    ax.add_patch(plt.Circle((yx[1] - sl[1].start, yx[0] - sl[0].start), r, fill=False, ec=col, lw=3))


def merge_overview(rgb, ph, cases):
    """Whole image, before vs after the neck-ratio merge, with the zoom boxes 1-3 of the check figure."""
    fig, ax = plt.subplots(2, 2, figsize=(20, 13.5))
    for r, ((final, st, boxes), (code, name)) in enumerate(zip(cases, [(GRAIN, "grains"), (PORE, "pores")])):
        mask = ph == code
        ws = st["watershed"]
        lut = np.random.default_rng(1).uniform(0.25, 1.0, (ws.max() + 1, 3))
        lut[0] = 0
        piece = ndi.maximum(ws, final, index=np.arange(final.max() + 1)).astype(int)   # merged pieces share a colour
        ws_cut, fin_cut = cuts(ws), cuts(final)
        undone = ndi.binary_dilation(ws_cut & ~fin_cut) & mask
        before = lut[ws].copy()
        before[ndi.binary_dilation(ws_cut) & mask] = (1, 1, 1)
        before[undone] = (1, 0.9, 0)
        after = lut[piece][final].copy()
        after[ndi.binary_dilation(fin_cut) & mask] = (1, 1, 1)
        n_merged = ws.max() - final.max()
        for c, (img, title) in enumerate([
                (before, f"BEFORE the merge (watershed only): {ws.max()} {name}\n"
                         f"white = cuts between pieces, YELLOW = cuts the merge will undo"),
                (after, f"AFTER the merge: {final.max()} {name}  ({n_merged} pieces joined back)\n"
                        f"white = cuts kept (real contacts / throats)")]):
            a = ax[r, c]
            a.imshow(img, interpolation="nearest")
            a.set_title(title, fontsize=12, loc="left")
            a.set_xticks([])
            a.set_yticks([])
            for num, yx in boxes:
                sl = crop_around(yx, rgb.shape[:2])
                a.add_patch(plt.Rectangle((sl[1].start, sl[0].start), CW, CH, fill=False, ec="red", lw=2.5))
                a.text(sl[1].start + 3, sl[0].start - 6, str(num), color="red", fontsize=16, fontweight="bold",
                       path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        ax[r, 0].set_ylabel(name.upper(), fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_OVERVIEW, dpi=85, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT_OVERVIEW)


def main():
    rgb = load_rgb(IMAGE)
    ph = np.load(PHASES)
    grains, gst = split_objects(ph == GRAIN, return_steps=True)
    pores, pst = split_objects(ph == PORE, return_steps=True)

    g_kept = cuts(gst["watershed"]) & cuts(grains)
    g_undone = cuts(gst["watershed"]) & ~cuts(grains) & (ph == GRAIN)
    p_kept = cuts(pst["watershed"]) & cuts(pores)

    # case 1: the longest kept cut between two grains of similar size
    yx1, _ = biggest_segment(g_kept)
    # case 2: the undone cut inside the largest grain
    area = ndi.sum(np.ones_like(grains), grains, index=np.arange(grains.max() + 1))
    weight = g_undone * area[grains]
    yx2, _ = biggest_segment(g_undone, weight)
    # case 3: the longest kept cut in the pore space (a throat)
    yx3, _ = biggest_segment(p_kept)

    merge_overview(rgb, ph, [(grains, gst, [(1, yx1), (2, yx2)]), (pores, pst, [(3, yx3)])])

    fig, ax = plt.subplots(3, 3, figsize=(17, 16))
    rows = [
        ("1. Touching grains must be SEPARATED", yx1,
         [("original photo", lambda s: rgb[s]),
          ("result: one colour per grain", lambda s: overlay(rgb[s], grains[s])),
          (None, None)]),
        ("2. A large elongated grain must NOT be cut", yx2,
         [("original photo", lambda s: rgb[s]),
          ("BEFORE the merge: wrongly cut in two", lambda s: overlay(rgb[s], gst["watershed"][s])),
          ("AFTER the merge: one grain (correct)", lambda s: overlay(rgb[s], grains[s]))]),
        ("3. Pores must be split at their THROATS", yx3,
         [("original photo", lambda s: rgb[s]),
          ("result: one colour per pore body", lambda s: overlay(rgb[s], pores[s], alpha=0.55, dark=0.7)),
          (None, None)]),
    ]
    notes = {
        0: "Look inside the circle: the two grains\ntouch, but each has its own colour and\n"
           "a white line separates them.\nThe neck between them is narrow\n(neck / width < 0.7), so the cut is kept.",
        2: "Look inside the circle: the blue pore\nnetwork narrows to a thin throat.\nThe split cuts it there, so each side\n"
           "becomes its own pore body\n(different colours).",
    }
    for r, (title, yx, panels) in enumerate(rows):
        sl = crop_around(yx, rgb.shape[:2])
        for c, (sub, fn) in enumerate(panels):
            a = ax[r, c]
            a.set_xticks([])
            a.set_yticks([])
            if fn is None:
                a.axis("off")
                a.text(0.02, 0.5, notes[r], fontsize=13, va="center", transform=a.transAxes,
                       bbox=dict(facecolor="#f4f4f4", edgecolor="0.6", boxstyle="round,pad=0.8"))
                continue
            a.imshow(fn(sl), interpolation="nearest")
            circle(a, yx, sl, col="yellow" if c else "red")
            a.set_title(sub, fontsize=12)
        ax[r, 0].set_ylabel(title, fontsize=14, fontweight="bold", labelpad=12)
    ax[1, 1].set_title("BEFORE the merge: wrongly cut in two", fontsize=12, color="firebrick", fontweight="bold")
    ax[1, 2].set_title("AFTER the merge: one grain (correct)", fontsize=12, color="green", fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=90, bbox_inches="tight")
    print("saved", OUT, "spots:", [tuple(int(v) for v in p) for p in (yx1, yx2, yx3)])


if __name__ == "__main__":
    main()
