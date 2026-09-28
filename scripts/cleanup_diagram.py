"""Explanatory figure for the morphological clean-up and the cement class (notebook 01, section 2).

(a) original crop
(b) phase map BEFORE clean-up (raw rules)
(c) phase map AFTER clean-up (as used in the project)
(d) what changed: holes filled inside grains, grain specks removed, pore specks removed
The crop is chosen automatically where the clean-up changes the most pixels.

Run from the project root:  python scripts/cleanup_diagram.py
Output: outputs/figures/cleanup_choice.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from segmentation import GRAIN, PORE, load_rgb, phase_rgb, three_phase  # noqa: E402

IMAGE = os.path.join(ROOT, "Lamina_alunos2.png")
OUT = os.path.join(ROOT, "outputs", "figures", "cleanup_choice.png")
MIN_SIZE = 20
CH, CW = 70, 100                                             # crop size (px)
CHANGE_COLOURS = {"hole filled -> grain": (0.1, 0.8, 0.1),
                  "grain speck removed -> cement": (1.0, 0.55, 0.0),
                  "pore speck removed -> cement": (0.9, 0.0, 0.9)}


def main():
    rgb = load_rgb(IMAGE)
    raw = three_phase(rgb, min_size=0)                        # rules only, nothing removed
    clean = three_phase(rgb, min_size=MIN_SIZE)               # as used in the project
    g0, g1 = raw.phases == GRAIN, clean.phases == GRAIN
    p0, p1 = raw.phases == PORE, clean.phases == PORE
    changes = {"hole filled -> grain": ~g0 & g1,
               "grain speck removed -> cement": g0 & ~g1,
               "pore speck removed -> cement": p0 & ~p1 & ~g1}
    any_change = np.logical_or.reduce(list(changes.values()))

    # crop where the clean-up changes the most pixels
    density = ndi.uniform_filter(any_change.astype(float), size=(CH, CW))
    cy, cx = np.unravel_index(np.argmax(density[CH // 2:-CH // 2, CW // 2:-CW // 2]), density[CH // 2:-CH // 2, CW // 2:-CW // 2].shape)
    sl = (slice(cy, cy + CH), slice(cx, cx + CW))

    n_objects = {k: ndi.label(v)[1] for k, v in changes.items()}

    fig, ax = plt.subplots(1, 4, figsize=(21, 5.8))
    ax[0].imshow(rgb[sl], interpolation="nearest")
    ax[0].set_title("(a) original (zoomed crop)", fontsize=12, loc="left")
    ax[1].imshow(phase_rgb(raw.phases)[sl], interpolation="nearest")
    ax[1].set_title("(b) BEFORE clean-up: rules only\n     note the small specks and holes", fontsize=12, loc="left")
    ax[2].imshow(phase_rgb(clean.phases)[sl], interpolation="nearest")
    ax[2].set_title(f"(c) AFTER clean-up (<= {MIN_SIZE} px removed / filled)", fontsize=12, loc="left")

    show = phase_rgb(clean.phases)[sl].astype(float) / 255 * 0.35 + 0.65       # faded background
    for k, col in CHANGE_COLOURS.items():
        show[changes[k][sl]] = col
    ax[3].imshow(show, interpolation="nearest")
    ax[3].set_title("(d) pixels changed by the clean-up", fontsize=12, loc="left")
    ax[3].legend(handles=[Patch(color=col, label=f"{k}  ({n_objects[k]} in whole image)")
                          for k, col in CHANGE_COLOURS.items()],
                 loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=10, frameon=False)
    ax[1].legend(handles=[Patch(color=np.array(c) / 255, label=n) for n, c in
                          [("cement", (90, 50, 30)), ("grain", (235, 230, 215)), ("pore", (30, 80, 200))]],
                 loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=10, frameon=False)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=90, bbox_inches="tight")
    print("saved", OUT, "crop", sl)
    print("objects changed (whole image):", n_objects)
    print("pixels changed:", {k: int(v.sum()) for k, v in changes.items()}, "of", any_change.size)
    print("fractions before:", {k: round(100 * v, 2) for k, v in raw.fractions.items()})
    print("fractions after: ", {k: round(100 * v, 2) for k, v in clean.fractions.items()})


if __name__ == "__main__":
    main()
