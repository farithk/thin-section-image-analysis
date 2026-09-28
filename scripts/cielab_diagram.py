"""Linked-view figure of the CIELAB colour space used in notebook 01.

Each row pairs a concept panel (left) with the thin-section data (right):
  row 1  brightness:          L* grey scale                     <->  L* histogram stacked by phase
  row 2  colour + brightness: a*-b* colour plane, hue & chroma  <->  pixel density on the L*-b* plane
  row 3  colour direction:    a*-b* plane (pixels)              <->  hue-chroma plot of the same pixels
Shared colours and reference lines (hue -80 / +40 deg, chroma 10 / 20 / 30) link the panels of row 3.

Run from the project root:  python scripts/cielab_diagram.py
Output: outputs/figures/cielab_linked_views.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import lab2rgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from segmentation import CEMENT, GRAIN, PORE, load_rgb, three_phase  # noqa: E402

IMAGE = os.path.join(ROOT, "Lamina_alunos2.png")
OUT = os.path.join(ROOT, "outputs", "figures", "cielab_linked_views.png")

PHASE_STYLE = {GRAIN: ("grain", "#d9a441"), PORE: ("pore (blue epoxy)", "#1f5fbf"), CEMENT: ("cement", "#8b4513")}
HUE_REF = {"pores": (-80, "#1f5fbf"), "cement": (40, "#8b4513")}
CHROMA_RINGS = (10, 20, 30)
HALO = [pe.withStroke(linewidth=4, foreground="white")]


def lab_image(L, a, b):
    return np.clip(lab2rgb(np.dstack([L, a, b])), 0, 1)


def colour_plane(ax, L_level=65, lim=85):
    """The a*-b* colour plane at one lightness, with hue (angle) and chroma (distance) explained."""
    A, B = np.meshgrid(np.linspace(-lim, lim, 341), np.linspace(-lim, lim, 341))
    ax.imshow(lab_image(np.full_like(A, L_level), A, B), extent=[-lim, lim, -lim, lim], origin="lower")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8)

    for (x, y), txt in [((58, 4), "+a* red"), ((-80, -6), "-a* green"), ((3, 62), "+b* yellow"), ((3, -66), "-b* blue")]:
        ax.text(x, y, txt, fontsize=10, fontweight="bold", path_effects=HALO)
    ax.text(2, 2, "grey\n(C = 0)", fontsize=9, path_effects=HALO)

    # example: the same hue at two chroma values
    h = -80
    for C, lw, lbl, xy in [(50, 3, "chroma C = 50\n= LENGTH: how colourful", (-44, -40)),
                           (15, 2, "same hue, C = 15:\nsame colour, but duller (greyer)", (13, -24))]:
        pa, pb = C * np.cos(np.radians(h)), C * np.sin(np.radians(h))
        ax.annotate("", xy=(pa, pb), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=lw, color="k", mutation_scale=16))
        ax.plot(pa, pb, "o", color=lab_image(np.array([[L_level]]), np.array([[pa]]), np.array([[pb]]))[0, 0],
                ms=11, mec="k", zorder=5)
        ax.text(*xy, lbl, fontsize=10, fontweight="bold", path_effects=HALO)
    arc = np.radians(np.linspace(0, h, 60))
    ax.plot(11 * np.cos(arc), 11 * np.sin(arc), color="k", lw=2)
    ax.text(14, -8, f"hue h = {h}°\n= ANGLE: which colour", fontsize=10, fontweight="bold", path_effects=HALO)
    ax.text(-82, 72, "h = atan2(b*, a*)\nC = sqrt(a*² + b*²)", fontsize=10, family="monospace",
            bbox=dict(facecolor="white", edgecolor="0.6"))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("a*  (- green  |  + red)")
    ax.set_ylabel("b*  (- blue  |  + yellow)")


def main():
    r = three_phase(load_rgb(IMAGE))
    lab, ph = r.lab.reshape(-1, 3), r.phases.ravel()
    L, a, b = lab.T
    hue, chroma = np.degrees(np.arctan2(b, a)), np.hypot(a, b)

    rng = np.random.default_rng(0)
    ng = np.flatnonzero(ph != GRAIN)
    idx = rng.choice(ng, 30000, replace=False)
    idx = idx[np.argsort(L[idx])]                                   # dark first, bright on top
    px_rgb = np.clip(lab2rgb(lab[idx][None])[0], 0, 1)             # each pixel's real colour

    fig, ax = plt.subplots(3, 2, figsize=(16, 19), gridspec_kw={"width_ratios": [1, 1.25]})

    # ---------------- row 1: brightness only
    a0 = ax[0, 0]
    Lbar = np.linspace(0, 100, 256)[:, None] * np.ones((1, 30))
    a0.imshow(lab_image(Lbar, 0 * Lbar, 0 * Lbar), extent=[0, 1, 0, 100], origin="lower", aspect="auto")
    for code, (name, c) in PHASE_STYLE.items():
        m = np.median(L[ph == code])
        dy = {GRAIN: 0, PORE: -8, CEMENT: 8}[code]
        a0.annotate(f"{name}: L* = {m:.0f}", xy=(1, m), xytext=(1.08, m + dy), va="center", color=c, fontsize=11,
                    fontweight="bold", arrowprops=dict(arrowstyle="->", color=c, lw=2), annotation_clip=False)
    a0.set_xlim(0, 1)
    a0.set_xticks([])
    a0.set_ylabel("L* (lightness)")
    a0.set_title("1a  CONCEPT: brightness L*, from black (0) to white (100)", loc="left", fontsize=12)

    a1 = ax[0, 1]
    bins = np.linspace(0, 100, 101)
    stack = [L[ph == code] for code in (CEMENT, PORE, GRAIN)]
    a1.hist(stack, bins=bins, orientation="horizontal", stacked=True,
            color=[PHASE_STYLE[c][1] for c in (CEMENT, PORE, GRAIN)],
            label=[PHASE_STYLE[c][0] for c in (CEMENT, PORE, GRAIN)])
    a1.axhline(r.t_L, color="k", ls="--", lw=1.5)
    a1.text(0.97, (r.t_L - 5) / 100, f"Otsu threshold L* = {r.t_L:.1f}", transform=a1.transAxes, ha="right", fontsize=10)
    a1.set_ylim(0, 100)
    a1.set_xlabel("number of pixels")
    a1.set_ylabel("L*")
    a1.legend(loc="upper right", bbox_to_anchor=(1, 0.75))
    a1.set_title("1b  DATA: L* histogram, coloured by final phase\n"
                 "     (same axis as 1a; the grayscale histogram looks the same)", loc="left", fontsize=12)
    a1.text(0.97, 0.3, "pore and cement fall in the SAME dark peak:\nbrightness cannot separate them",
            transform=a1.transAxes, fontsize=11, color="k", ha="right")

    # ---------------- row 2: the colour plane (concept) and brightness x blue-yellow (data)
    a2 = ax[1, 0]
    colour_plane(a2)
    a2.set_title("2a  CONCEPT: the a*-b* colour plane (grey in the centre)\n"
                 "     hue = ANGLE (which colour), chroma = DISTANCE (how colourful)", loc="left", fontsize=12)

    a3 = ax[1, 1]
    Lg, bg = np.meshgrid(np.linspace(0, 100, 201), np.linspace(-50, 20, 141))
    a3.imshow(lab_image(Lg, 0 * Lg, bg), extent=[0, 100, -50, 20], origin="lower", aspect="auto", alpha=0.45)
    a3.axhline(0, color="k", lw=0.6)
    H, xe, ye = np.histogram2d(L, b, bins=[200, 140], range=[[0, 100], [-50, 20]])
    Hm = np.ma.masked_where(H == 0, np.log10(H + 1))
    a3.pcolormesh(xe, ye, Hm.T, cmap="Greys", shading="auto")
    a3.axvline(r.t_L, color="k", ls="--", lw=1.5)
    for txt, tip, xy, c in [("grains (bright, neutral)", (80, 4), (58, 15), PHASE_STYLE[GRAIN][1]),
                            ("pores: dark and blue", (17, -30), (30, -38), PHASE_STYLE[PORE][1]),
                            ("cement: dark, neutral", (4, 0), (8, 15), PHASE_STYLE[CEMENT][1]),
                            ("mixed pixels at\ngrain edges", (35, 1), (52, -14), "k")]:
        a3.annotate(txt, xy=tip, xytext=xy, fontsize=11, fontweight="bold", color=c,
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.5))
    a3.set_xlabel("L* (lightness)  ->  brighter")
    a3.set_ylabel("b*  (- blue  |  + yellow)")
    a3.set_title("2b  DATA: pixel density on the L*-b* plane (log scale)\n"
                 "     L-shape: each arm = one phase; dashed = grain threshold", loc="left", fontsize=12)

    # ---------------- row 3: colour direction (hue) and strength (chroma)
    a4 = ax[2, 0]
    ag, bgg = np.meshgrid(np.linspace(-35, 35, 281), np.linspace(-50, 20, 281))
    a4.imshow(lab_image(60 + 0 * ag, ag, bgg), extent=[-35, 35, -50, 20], origin="lower", alpha=0.35)
    a4.scatter(a[idx], b[idx], c=px_rgb, s=3, edgecolors="none")
    tt = np.radians(np.linspace(-180, 180, 300))
    for cr in CHROMA_RINGS:
        a4.plot(cr * np.cos(tt), cr * np.sin(tt), color="0.35", ls=":", lw=1)
        a4.text(cr * np.cos(np.radians(-20)), cr * np.sin(np.radians(-20)), f"C={cr}", fontsize=9, color="0.25")
    for name, (h, c) in HUE_REF.items():
        a4.plot([0, 60 * np.cos(np.radians(h))], [0, 60 * np.sin(np.radians(h))], color=c, ls="--", lw=2)
        a4.text(*{"pores": (-30, -30), "cement": (3, 15)}[name], f"{name}\nh = {h:+d} deg",
                color=c, fontsize=11, fontweight="bold")
    a4.axhline(0, color="k", lw=0.5)
    a4.axvline(0, color="k", lw=0.5)
    a4.set_xlim(-35, 35)
    a4.set_ylim(-50, 20)
    a4.set_aspect("equal")
    a4.set_xlabel("a*  (- green  |  + red)")
    a4.set_ylabel("b*  (- blue  |  + yellow)")
    a4.set_title("3a  CONCEPT + DATA: dark pixels on the a*-b* plane\n"
                 "     direction = hue h,  distance from centre = chroma C", loc="left", fontsize=12)

    a5 = ax[2, 1]
    hg, cg = np.meshgrid(np.linspace(-180, 180, 361), np.linspace(0, 50, 101))
    a5.imshow(lab_image(60 + 0 * hg, cg * np.cos(np.radians(hg)), cg * np.sin(np.radians(hg))),
              extent=[-180, 180, 0, 50], origin="lower", aspect="auto", alpha=0.35)
    a5.scatter(hue[idx], chroma[idx], c=px_rgb, s=3, edgecolors="none")
    for cr in CHROMA_RINGS:
        a5.axhline(cr, color="0.35", ls=":", lw=1)
        a5.text(-178, cr + 0.6, f"C={cr}", fontsize=9, color="0.25")
    for name, (h, c) in HUE_REF.items():
        a5.axvline(h, color=c, ls="--", lw=2)
        a5.text(h + 4, 44, f"{name}\nh = {h:+d} deg", color=c, fontsize=11, fontweight="bold")
    a5.text(95, 2, "near-grey pixels:\nhue is random", fontsize=10)
    a5.set_xlim(-180, 180)
    a5.set_ylim(0, 50)
    a5.set_xticks(range(-180, 181, 45))
    a5.set_xlabel("hue angle h (deg)  = direction in 3a")
    a5.set_ylabel("chroma C  = distance from centre in 3a")
    a5.set_title("3b  DATA: the SAME pixels, unrolled (angle -> x, distance -> y)\n"
                 "     dashed lines = dashed rays in 3a; dotted = rings in 3a", loc="left", fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=90, bbox_inches="tight")
    print("saved", OUT)


if __name__ == "__main__":
    main()
