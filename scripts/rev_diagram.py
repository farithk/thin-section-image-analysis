"""Plain-language figure for the REV (notebook 04, section 1).

Row 1: the large image with random windows of three sizes, each labelled with the porosity measured inside it.
Row 2: every window's porosity as a dot, for many window sizes; the band = "close" to the whole-image porosity
       (within +-10 %), and the number of close windows out of 100 for each size.

Run from the project root:  python scripts/rev_diagram.py
Output: outputs/figures/rev_simple.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from segmentation import PORE, load_rgb  # noqa: E402

IMAGE = os.path.join(ROOT, "Lamina_alunos_grande.png")
PHASES = os.path.join(ROOT, "outputs", "phases_large.npy")
OUT = os.path.join(ROOT, "outputs", "figures", "rev_simple.png")
N = 100


def window_porosity(ii, y, x, L):
    return (ii[y + L, x + L] - ii[y, x + L] - ii[y + L, x] + ii[y, x]) / L ** 2


def main():
    rgb = load_rgb(IMAGE)
    pore = (np.load(PHASES) == PORE).astype(float)
    H, W = pore.shape
    ii = np.zeros((H + 1, W + 1))
    ii[1:, 1:] = pore.cumsum(0).cumsum(1)
    ref = pore.mean()
    lo, hi = 0.9 * ref, 1.1 * ref
    rng = np.random.default_rng(3)

    fig = plt.figure(figsize=(19, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.28, wspace=0.08)

    # row 1: windows on the image
    for c, (L, n_show) in enumerate([(50, 8), (200, 5), (500, 3)]):
        a = fig.add_subplot(gs[0, c])
        a.imshow(rgb)
        ys, xs = rng.integers(0, H - L + 1, n_show), rng.integers(0, W - L + 1, n_show)
        vals = window_porosity(ii, ys, xs, L)
        for y, x, v in zip(ys, xs, vals):
            ok = lo <= v <= hi
            col = "lime" if ok else "red"
            a.add_patch(plt.Rectangle((x, y), L, L, fill=False, ec=col, lw=2.5))
            a.text(x + 4, y + 4, f"{100 * v:.0f}%", color="white", fontsize=12 if L > 50 else 10, fontweight="bold",
                   va="top", bbox=dict(facecolor=col, alpha=0.75, edgecolor="none", pad=1.5))
        a.set_title(f"windows of {L} px: porosity in each one\n"
                    f"{'very different answers' if L == 50 else 'closer answers' if L == 200 else 'almost the same answer'}",
                    fontsize=13)
        a.axis("off")

    # row 2: dots for every window size
    a = fig.add_subplot(gs[1, :])
    sizes = [25, 50, 75, 100, 150, 200, 275, 360, 450, 550]
    for k, L in enumerate(sizes):
        ys, xs = rng.integers(0, H - L + 1, N), rng.integers(0, W - L + 1, N)
        v = 100 * window_porosity(ii, ys, xs, L)
        close = ((v >= 100 * lo) & (v <= 100 * hi)).sum()
        jitter = rng.uniform(-0.25, 0.25, N)
        cols = np.where((v >= 100 * lo) & (v <= 100 * hi), "green", "red")
        a.scatter(k + jitter, v, s=14, c=cols, alpha=0.7)
        a.text(k, 72, f"{close}\nof 100", ha="center", fontsize=11, fontweight="bold",
               color="green" if close >= 95 else "black")
    a.axhspan(100 * lo, 100 * hi, color="green", alpha=0.15)
    a.axhline(100 * ref, color="green", lw=1.5)
    a.text(len(sizes) - 0.45, 100 * ref + 2.5, f"whole-image porosity {100 * ref:.1f}%\n"
                                                f"green band = 'close' ({100 * lo:.1f}-{100 * hi:.1f}%)",
           color="green", fontsize=11, ha="right", va="bottom")
    a.text(-0.6, 72, "windows\nin the band:", fontsize=11, ha="right", fontweight="bold")
    a.set_xticks(range(len(sizes)))
    a.set_xticklabels([f"{L} px" for L in sizes])
    a.set_xlim(-1.4, len(sizes) - 0.4)
    a.set_ylim(-2, 80)
    a.set_xlabel("window size (side of the square, in pixels)", fontsize=12)
    a.set_ylabel("porosity measured in the window (%)", fontsize=12)
    a.set_title("100 random windows per size: each dot = one window (green = close to the whole-image porosity, "
                "red = far).\nSmall windows scatter from 0 to 70 %; bigger windows squeeze into the green band. "
                "REV = the size where almost all (95 of 100) are green.", fontsize=12, loc="left")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=80, bbox_inches="tight")
    print("saved", OUT)


if __name__ == "__main__":
    main()
