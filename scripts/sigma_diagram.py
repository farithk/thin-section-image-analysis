"""Explanatory figure for the choice of the Gaussian smoothing sigma (notebook 01, section 2).

Row 1: a zoomed crop smoothed with increasing sigma (noise removed, then edges blurred)
Row 2: pore classification (red) on the same crop for each sigma
Row 3: porosity, noise (tiny pore fragments) and edge blur width as a function of sigma

Run from the project root:  python scripts/sigma_diagram.py
Output: outputs/figures/sigma_choice.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import filters

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from segmentation import GRAIN, PORE, load_rgb, three_phase  # noqa: E402

IMAGE = os.path.join(ROOT, "Lamina_alunos2.png")
OUT = os.path.join(ROOT, "outputs", "figures", "sigma_choice.png")
CROP = (slice(255, 345), slice(300, 435))          # dark pores and thin throats (small image)
SIGMAS_SHOWN = (0, 1, 2, 3)
SIGMAS_SCAN = (0, 0.5, 1, 1.5, 2, 2.5, 3)
CHOSEN = 1


def edge_width(rgb, sigma):
    """Mean 10-90 % width (px) of grain edges: L* contrast / gradient magnitude at the grain boundary."""
    r = three_phase(rgb, sigma=sigma)
    L = r.lab[..., 0]
    grad = np.hypot(ndi.sobel(L, 0), ndi.sobel(L, 1)) / 8.0                  # Sobel -> L* units per px
    boundary = filters.sobel(r.phases == GRAIN) > 0
    contrast = np.median(L[r.phases == GRAIN]) - np.median(L[r.phases != GRAIN])
    return 0.8 * contrast / np.median(grad[boundary])


def main():
    rgb = load_rgb(IMAGE)
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.5])

    # rows 1-2: the crop and its pore classification for each sigma
    for j, s in enumerate(SIGMAS_SHOWN):
        r = three_phase(rgb, sigma=s)
        sm = filters.gaussian(rgb, sigma=s, channel_axis=-1) if s > 0 else rgb
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(sm[CROP])
        ax.set_title(f"sigma = {s}" + ("   (chosen)" if s == CHOSEN else ""), fontsize=13,
                     fontweight="bold" if s == CHOSEN else "normal", color="tab:green" if s == CHOSEN else "k")
        ax.axis("off")
        ax = fig.add_subplot(gs[1, j])
        ov = rgb[CROP].copy()
        ov[r.phases[CROP] == PORE] = [1.0, 0.15, 0.15]
        ax.imshow(ov)
        ax.set_title(f"pore pixels (red): porosity {100 * r.fractions['pore']:.1f} %", fontsize=11)
        ax.axis("off")
    fig.text(0.005, 0.86, "1. image after\nsmoothing", fontsize=12, fontweight="bold", va="center")
    fig.text(0.005, 0.6, "2. pore rule\napplied", fontsize=12, fontweight="bold", va="center")

    # row 3: quantitative effect of sigma
    por, frag, width = [], [], []
    for s in SIGMAS_SCAN:
        r = three_phase(rgb, sigma=s, min_size=0)                            # no clean-up: count the raw specks
        lab, n = ndi.label(r.phases == PORE)
        sizes = np.bincount(lab.ravel())[1:]
        frag.append(int((sizes < 20).sum()))
        por.append(100 * three_phase(rgb, sigma=s).fractions["pore"])
        width.append(edge_width(rgb, s))

    ax = fig.add_subplot(gs[2, 0:1])
    ax.plot(SIGMAS_SCAN, frag, "o-", color="tab:red", lw=2)
    ax.axvline(CHOSEN, color="tab:green", ls="--", lw=2)
    ax.set_xlabel("sigma (px)")
    ax.set_ylabel("tiny pore fragments (< 20 px)")
    ax.set_title("NOISE: isolated specks\n(drops sharply up to sigma = 1)", fontsize=11)

    ax = fig.add_subplot(gs[2, 1:2])
    ax.plot(SIGMAS_SCAN, por, "o-", color="tab:blue", lw=2)
    ax.axvline(CHOSEN, color="tab:green", ls="--", lw=2)
    ax.set_xlabel("sigma (px)")
    ax.set_ylabel("porosity (%)")
    ax.set_title("POROSITY: rises quickly to sigma = 1,\nthen slowly (edges start to spread)", fontsize=11)

    ax = fig.add_subplot(gs[2, 2:4])
    ax.plot(SIGMAS_SCAN, width, "o-", color="k", lw=2, label="edge blur width (10-90 %)")
    for val, txt, c in [(3.5, "pore throats ~ 3-4 px", "tab:purple"), (11, "pore bodies ~ 11 px", "tab:blue"),
                        (14, "grains ~ 14 px", "goldenrod")]:
        ax.axhline(val, color=c, ls=":", lw=2)
        ax.text(SIGMAS_SCAN[0], val + 0.3, txt, color=c, ha="left", fontsize=10, fontweight="bold")
    ax.axvline(CHOSEN, color="tab:green", ls="--", lw=2, label=f"chosen sigma = {CHOSEN}")
    ax.set_xlabel("sigma (px)")
    ax.set_ylabel("width (px)")
    ax.set_ylim(0, 16)
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.8))
    ax.set_title("BLUR: edge width grows with sigma; it must stay well below the throat size", fontsize=11)

    fig.text(0.005, 0.22, "3. effect\nof sigma", fontsize=12, fontweight="bold", va="center")
    plt.tight_layout(rect=(0.07, 0, 1, 1))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=90, bbox_inches="tight")
    print("saved", OUT)
    for s, f, p, w in zip(SIGMAS_SCAN, frag, por, width):
        print(f"sigma {s}: fragments {f}, porosity {p:.2f} %, edge width {w:.2f} px")


if __name__ == "__main__":
    main()
