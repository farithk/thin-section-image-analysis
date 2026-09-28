"""Explanatory figure for the grain rule: L* > Otsu threshold (notebook 01, section 2).

(a) L* histogram split at the Otsu cut, with the mean of each group
(b) zoomed crop with grain outlines at three cuts (too low, Otsu, too high)

Run from the project root:  python scripts/otsu_diagram.py
Output: outputs/figures/otsu_choice.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters, measure

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from segmentation import load_rgb, to_lab  # noqa: E402

IMAGE = os.path.join(ROOT, "Lamina_alunos2.png")
OUT = os.path.join(ROOT, "outputs", "figures", "otsu_choice.png")
CROP = (slice(95, 185), slice(95, 230))
CUTS = {30: ("too low", "tab:orange"), None: ("Otsu", "tab:green"), 65: ("too high", "tab:purple")}


def main():
    L = to_lab(load_rgb(IMAGE))[..., 0]
    t_otsu = filters.threshold_otsu(L)
    mu_dark, mu_bright = L[L <= t_otsu].mean(), L[L > t_otsu].mean()

    fig, ax = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={"width_ratios": [1, 1.3]})

    # (a) histogram split at the cut
    a = ax[0]
    h, e = np.histogram(L, bins=100, range=(0, 100))
    c = 0.5 * (e[1:] + e[:-1])
    a.bar(c[c <= t_otsu], h[c <= t_otsu], width=1, color="0.3", label="below cut: pore + cement")
    a.bar(c[c > t_otsu], h[c > t_otsu], width=1, color="#d9a441", label="above cut: grain")
    a.axvline(t_otsu, color="tab:green", ls="--", lw=2.5)
    a.text(t_otsu + 1, h.max() * 0.9, f"Otsu cut\nL* = {t_otsu:.1f}", color="tab:green", fontweight="bold")
    for mu, txt in [(mu_dark, "mean of\ndark group"), (mu_bright, "mean of\nbright group")]:
        a.axvline(mu, color="k", ls=":", lw=1.5)
        a.text(mu, h.max() * 0.55, f"{txt}\n= {mu:.0f}", ha="center", fontsize=9,
               bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
    a.annotate("mixed pixels\nat grain edges", xy=(35, h[35] * 1.3), xytext=(22, h.max() * 0.3), fontsize=9,
               arrowprops=dict(arrowstyle="->"))
    a.set_xlabel("L* (brightness)")
    a.set_ylabel("number of pixels")
    a.legend(loc="upper left", fontsize=9)
    a.set_title("(a) the L* histogram is split into two groups at the Otsu cut", loc="left")

    # (b) outlines on a crop
    b = ax[1]
    crop = L[CROP]
    b.imshow(crop, cmap="gray", vmin=0, vmax=100)
    for t, (lbl, col) in CUTS.items():
        tt = t_otsu if t is None else t
        for cn in measure.find_contours(crop, tt):
            b.plot(cn[:, 1], cn[:, 0], color=col, lw=1.6)
        b.plot([], [], color=col, lw=2.5, label=f"cut L* = {tt:.1f} ({lbl}): grain {100 * (L > tt).mean():.1f} %")
    b.legend(loc="upper right", fontsize=10, framealpha=0.9)
    b.axis("off")
    b.set_title("(b) grain outlines at three cuts: too low -> grains too fat, too high -> too thin", loc="left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=90, bbox_inches="tight")
    print("saved", OUT, f"(Otsu t = {t_otsu:.2f})")


if __name__ == "__main__":
    main()
