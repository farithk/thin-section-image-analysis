"""Explanatory figure for the pore rule: hue in (-140, -40) deg and chroma > 5 (notebook 01, section 2).

HUE    (a) hue histogram of the dark pixels, each bar in the colour of its hue; pore range shaded
       (b) porosity for several hue ranges (the limits sit in empty zones, so they hardly matter)
CHROMA (c) swatches: the same blue / brown hue at increasing chroma, at the darkness of real pores
       (d) pore pixels (red) with chroma limits 3, 5, 8 on a zoomed crop

Run from the project root:  python scripts/pore_rule_diagram.py
Output: outputs/figures/pore_rule_choice.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import lab2rgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from segmentation import PORE, load_rgb, three_phase  # noqa: E402

IMAGE = os.path.join(ROOT, "Lamina_alunos2.png")
OUT = os.path.join(ROOT, "outputs", "figures", "pore_rule_choice.png")
CROP = (slice(255, 345), slice(300, 435))          # dark pores (small image)
HUE_RANGE = (-140, -40)
C_MIN = 5


def lab_colour(L, h_deg, C):
    h = np.radians(h_deg)
    return np.clip(lab2rgb(np.array([[[L, C * np.cos(h), C * np.sin(h)]]])), 0, 1)[0, 0]


def main():
    rgb = load_rgb(IMAGE)
    r = three_phase(rgb)
    L, a, b = r.lab[..., 0], r.lab[..., 1], r.lab[..., 2]
    hue, C = np.degrees(np.arctan2(b, a)), np.hypot(a, b)
    dark = L <= r.t_L

    fig = plt.figure(figsize=(19, 16))
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.55], hspace=0.04)
    top.suptitle("PART 1 - HUE (which colour): choosing -140 deg to -40 deg", fontsize=16, fontweight="bold", x=0.02, ha="left")
    bottom.suptitle("PART 2 - CHROMA (how much colour): choosing C > 5", fontsize=16, fontweight="bold", x=0.02, ha="left")
    gt = top.add_gridspec(1, 4, wspace=0.45, left=0.06, right=0.98, bottom=0.14, top=0.84)
    gb = bottom.add_gridspec(2, 4, height_ratios=[1.25, 1], hspace=0.35, wspace=0.3,
                             left=0.06, right=0.98, bottom=0.04, top=0.86)

    # (a) hue histogram, bars coloured by their hue
    ax = top.add_subplot(gt[0, 0:3])
    edges = np.arange(-180, 181, 5)
    cnt, _ = np.histogram(hue[dark & (C > C_MIN)], bins=edges)
    centers = 0.5 * (edges[1:] + edges[:-1])
    ax.bar(centers, np.maximum(cnt, 0.8), width=5, color=[lab_colour(50, h, 40) for h in centers], edgecolor="k", lw=0.3)
    ax.axvspan(*HUE_RANGE, facecolor="none", edgecolor="tab:blue", hatch="//", lw=2)
    ax.set_yscale("log")
    ax.set_ylim(0.8, cnt.max() * 8)
    ax.set_xlim(-180, 180)
    ax.set_xticks(range(-180, 181, 20))
    for x, y, txt in [(-80, cnt.max() * 2.2, "blue epoxy\n(pores)"), (30, cnt.max() * 0.25, "cement\n(brown)"),
                      (-22, 3e3, "nearly empty\nvalley"), (-158, 2, "empty")]:
        ax.text(x, y, txt, ha="center", fontsize=11, fontweight="bold")
    ax.text(-90, 1.3, f"pore range {HUE_RANGE[0]} to {HUE_RANGE[1]} deg", color="tab:blue", ha="center",
            fontsize=11, fontweight="bold", bbox=dict(facecolor="white", edgecolor="none"))
    ax.set_xlabel("hue angle (deg)   -   each bar is painted in the colour of its angle")
    ax.set_ylabel("number of dark pixels (log)")
    ax.set_title("(a) hue of the dark pixels: the blue peak is narrow and has empty zones on both sides; "
                 "the limits are placed in those empty zones", loc="left", fontsize=11)

    # (b) porosity vs hue range
    ax = top.add_subplot(gt[0, 3])
    ranges = [(-110, -60), (-120, -50), HUE_RANGE, (-160, -20), (-180, 0)]
    por = [100 * three_phase(rgb, hue_range=hr).fractions["pore"] for hr in ranges]
    cols = ["tab:green" if hr == HUE_RANGE else "0.6" for hr in ranges]
    ax.barh([f"{lo} to {hi}" for lo, hi in ranges], por, color=cols)
    for i, p in enumerate(por):
        ax.text(p + 0.2, i, f"{p:.1f} %", va="center", fontsize=10)
    ax.set_xlim(0, 24)
    ax.set_xlabel("porosity (%)")
    ax.set_title("(b) porosity for other hue ranges:\nalmost the same -> limits don't matter", loc="left", fontsize=11)

    # (c) swatches: what chroma looks like for a dark pixel
    ax = bottom.add_subplot(gb[0, 0:4])
    Cs = [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30]
    Ld = float(np.median(L[r.phases == PORE]))
    swatch_rows = [(50, -80, "blue, medium brightness"), (50, 40, "brown, medium brightness"),
                   (Ld, -80, f"blue, like a real pore (L* = {Ld:.0f})"), (Ld, 40, f"brown, like real cement (L* = {Ld:.0f})")]
    for row, (Lv, h, name) in enumerate(swatch_rows):
        y = 3 - row - (0.35 if row >= 2 else 0)
        for j, c in enumerate(Cs):
            ax.add_patch(plt.Rectangle((j, y), 0.92, 0.85, color=lab_colour(Lv, h, c)))
        ax.text(-0.2, y + 0.42, name, ha="right", va="center", fontsize=11, fontweight="bold")
    for j, c in enumerate(Cs):
        ax.text(j + 0.46, -0.65, f"C = {c}", ha="center", fontsize=10.5)
    x5 = Cs.index(C_MIN)
    ax.plot([x5 - 0.04, x5 - 0.04], [-0.4, 4.0], color="r", lw=3)
    ax.text(x5 / 2, 4.15, "C < 5: almost no colour -> blue and brown\nlook the same (grey / black)",
            ha="center", color="r", fontsize=11, fontweight="bold")
    ax.text((x5 + len(Cs)) / 2, 4.15, "C > 5: the colour becomes visible ->\nblue and brown can be told apart",
            ha="center", color="tab:green", fontsize=11, fontweight="bold")
    ax.set_xlim(-4.2, len(Cs))
    ax.set_ylim(-0.9, 5.0)
    ax.axis("off")
    ax.set_title("(c) what chroma means: the SAME two colours (blue and brown hue) painted with more and more chroma.\n"
                 "     Chroma = amount of colour. At 0 every hue is grey; the higher C, the more colourful.",
                 loc="left", fontsize=11)

    # (d) overlays at three chroma limits
    for j, cmin in enumerate([3, 5, 8]):
        rr = three_phase(rgb, min_chroma=cmin)
        ax = bottom.add_subplot(gb[1, j])
        ov = rgb[CROP].copy()
        ov[rr.phases[CROP] == PORE] = [1.0, 0.15, 0.15]
        ax.imshow(ov)
        verdict = {3: "too low: dark cement\ncounted as pore", 5: "chosen: matches the\nvisible blue epoxy",
                   8: "too high: dark blue\npores missed"}[cmin]
        ax.set_title(("(d) pore pixels in red\n" if j == 0 else "\n") + f"C > {cmin}: porosity {100 * rr.fractions['pore']:.1f} %\n{verdict}", fontsize=11,
                     color="tab:green" if cmin == C_MIN else "k", fontweight="bold" if cmin == C_MIN else "normal")
        ax.axis("off")
    ax = bottom.add_subplot(gb[1, 3])
    ax.imshow(rgb[CROP])
    ax.set_title("original crop (no overlay)", fontsize=11)
    ax.axis("off")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=85, bbox_inches="tight")
    print("saved", OUT)
    print("hue ranges:", dict(zip([str(x) for x in ranges], np.round(por, 2))))


if __name__ == "__main__":
    main()
