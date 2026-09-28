"""Plain-language figure for the block correlations (notebook 04, 'What controls porosity?').

(a) the large image cut into 88 squares of 128 Lpx; one square is highlighted with its measured values;
(b) porosity against cement, one dot per square (real data);
(c) the same data turned into ranks (1 = lowest): Spearman rho is the ordinary correlation of these ranks;
(d) what rho = +1, 0 and -1 look like (made-up dots).

Run from the project root:  python scripts/spearman_diagram.py
Output: outputs/figures/spearman_simple.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from segmentation import load_rgb  # noqa: E402

IMAGE = os.path.join(ROOT, "Lamina_alunos_grande.png")
BLOCKS = os.path.join(ROOT, "outputs", "04_blocks.csv")
OUT = os.path.join(ROOT, "outputs", "figures", "spearman_simple.png")
B = 128

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})


def main():
    rgb = load_rgb(IMAGE)
    bl = pd.read_csv(BLOCKS)
    por, cem = 100 * bl.porosity.values, 100 * bl.cement.values
    rho = stats.spearmanr(cem, por).statistic
    k = int(np.argmax(por))                                     # most porous square, as the example
    fig, ax = plt.subplots(2, 2, figsize=(15, 12))

    a = ax[0, 0]
    a.imshow(rgb)
    for x, y in zip(bl.bx, bl.by):
        a.add_patch(Rectangle((x, y), B, B, fill=False, ec="white", lw=0.8))
    a.add_patch(Rectangle((bl.bx[k], bl.by[k]), B, B, fill=False, ec="red", lw=3))
    a.text(bl.bx[k] + B / 2, bl.by[k] - 8, f"one square:\nporosity {por[k]:.0f} %\ncement {cem[k]:.0f} %",
           color="red", ha="center", va="bottom", fontsize=11, fontweight="bold",
           bbox=dict(fc="white", ec="red", alpha=0.9))
    a.set_title(f"(a) the image cut into {len(bl)} squares of {B} Lpx;\nin each square every property is measured")
    a.axis("off")

    a = ax[0, 1]
    a.scatter(cem, por, s=35, c="tab:blue", ec="k", lw=0.4)
    a.scatter(cem[k], por[k], s=120, facecolor="none", ec="red", lw=2.5)
    a.set_xlabel("cement in the square (%)"); a.set_ylabel("porosity in the square (%)")
    a.set_title("(b) one dot per square: more cement, less porosity")
    a.grid(alpha=0.3)

    rc, rp = stats.rankdata(cem), stats.rankdata(por)
    a = ax[1, 0]
    a.scatter(rc, rp, s=35, c="tab:orange", ec="k", lw=0.4)
    a.scatter(rc[k], rp[k], s=120, facecolor="none", ec="red", lw=2.5)
    a.set_xlabel(f"cement rank (1 = least cement ... {len(bl)} = most)")
    a.set_ylabel(f"porosity rank (1 = least porous ... {len(bl)} = most)")
    a.set_title(f"(c) values replaced by their rank: Spearman rho = {rho:.2f}\n"
                "(the trend is measured on the order, not on the values)")
    a.grid(alpha=0.3)

    a = ax[1, 1]
    names = {"porosity": "porosity", "cement": "cement", "grain": "grain fraction", "pore_D50": "pore size",
             "grain_D50": "grain size", "grain_sorting_phi": "grain sorting", "grain_circularity": "grain roundness"}
    v = list(names)
    n = len(v)
    rho_m = np.array([[stats.spearmanr(bl[x], bl[y], nan_policy="omit").statistic for y in v] for x in v])
    p_m = np.array([[stats.spearmanr(bl[x], bl[y], nan_policy="omit").pvalue for y in v] for x in v])
    shown = rho_m.copy()
    np.fill_diagonal(shown, np.nan)                             # a property with itself is always 1: greyed out
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("0.85")
    im = a.imshow(shown, cmap=cmap, vmin=-1, vmax=1)
    for i in range(n):
        for j in range(n):
            if i != j:
                a.text(j, i, f"{rho_m[i, j]:+.2f}" + ("*" if p_m[i, j] < 0.01 else ""), ha="center", va="center",
                       fontsize=10, color="white" if abs(rho_m[i, j]) > 0.5 else "black")
    a.add_patch(Rectangle((-0.5, -0.5), n, 1, fill=False, ec="k", lw=2.5))       # the porosity row
    a.set_xticks(range(n)); a.set_xticklabels([names[x] for x in v], rotation=40, ha="right")
    a.set_yticks(range(n)); a.set_yticklabels([names[x] for x in v])
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.04,
                 label="Spearman rho  (red = go up together, blue = one up, other down)")
    a.set_title("(d) rho for every pair of properties (* = real effect, p < 0.01);\n"
                "framed row = correlations with porosity (Table 9)")

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print("saved", OUT, "rho =", round(rho, 2))


if __name__ == "__main__":
    main()
