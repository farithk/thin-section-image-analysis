"""One explanatory image per step of the object separation (report section 4.2), on simple made-up shapes.

step1_distance.png   distance map: thick centre = hill, thin contact = valley
step2_markers.png    markers on the hill tops; the h rule ignores tiny bumps
step3_watershed.png  flooding from the markers; the boundary forms where the waters meet (at the neck)
step4_neck_ratio.png two touching grains (thin neck, cut kept) vs one grain with a waist (wide neck, cut undone)

Run from the project root:  python scripts/watershed_steps_diagram.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology, segmentation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "figures")
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13})
N = 120
yy, xx = np.mgrid[:N // 2 + 20, :N]
CY = (N // 2 + 20) // 2


def disks(centres, r):
    m = np.zeros(yy.shape, bool)
    for cx in centres:
        m |= (yy - CY) ** 2 + (xx - cx) ** 2 <= r * r
    return m


TWO = disks([38, 82], 23)                                  # two grains touching at a thin neck
WAIST = disks([42, 78], 24) | ((np.abs(yy - CY) <= 19) & (xx >= 42) & (xx <= 78))   # one grain with a slight waist


def cuts(labels):
    b = np.zeros(labels.shape, bool)
    b[:, 1:] |= (labels[:, 1:] != labels[:, :-1]) & (labels[:, 1:] > 0) & (labels[:, :-1] > 0)
    b[1:, :] |= (labels[1:, :] != labels[:-1, :]) & (labels[1:, :] > 0) & (labels[:-1, :] > 0)
    return b


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def step1():
    dt = ndi.distance_transform_edt(TWO)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1, 1.15, 1.3]})
    ax[0].imshow(TWO, cmap="Oranges", vmin=0, vmax=1.6)
    ax[0].set_title("(a) the phase map: two touching\ngrains = ONE connected area")
    ax[0].axhline(CY, color="k", ls="--", lw=1)
    im = ax[1].imshow(np.ma.masked_where(~TWO, dt), cmap="magma")
    cb = plt.colorbar(im, ax=ax[1], fraction=0.04)
    cb.ax.set_title("px", fontsize=11)
    ax[1].axhline(CY, color="c", ls="--", lw=1.5)
    ax[1].set_title("(b) distance map: bright = far from\nthe edge (thick), dark = near (thin)")
    prof = dt[CY]
    x = np.arange(N)
    ax[2].fill_between(x, 0, prof, color="#e8dcc0")
    ax[2].plot(x, prof, "k", lw=2)
    k = int(np.argmin(np.where((x > 45) & (x < 75), prof, 99)))
    for cx, lbl in [(38, "centre of\ngrain A"), (82, "centre of\ngrain B")]:
        ax[2].annotate(lbl, xy=(cx, prof[cx]), xytext=(cx, prof[cx] + 5), ha="center",
                       arrowprops=dict(arrowstyle="->"))
    ax[2].annotate("contact\n(narrow neck)", xy=(k, prof[k]), xytext=(k, prof[k] + 12), ha="center", color="firebrick",
                   arrowprops=dict(arrowstyle="->", color="firebrick"))
    ax[2].set_ylim(0, prof.max() + 12)
    ax[2].set_xlabel("position along the dashed line (px)")
    ax[2].set_ylabel("distance to edge (px)")
    ax[2].set_title("(c) along the dashed line: two HILLS\n(grain centres) and a VALLEY (contact)")
    for a in ax[:2]:
        a.set_xticks([])
        a.set_yticks([])
    fig.subplots_adjust(wspace=0.45)
    save(fig, "step1_distance.png")


def step2():
    rough = TWO.copy()
    rng = np.random.default_rng(3)
    edge = TWO & ~ndi.binary_erosion(TWO)
    ey, ex = np.nonzero(edge)
    pick = rng.choice(len(ey), 40, replace=False)
    rough[ey[pick], ex[pick]] = False                     # a rough, pixelated outline
    dt = ndi.gaussian_filter(ndi.distance_transform_edt(rough), 1.0)
    real = morphology.h_maxima(dt, 0.5) & rough
    allmax = (dt == ndi.maximum_filter(dt, size=3)) & rough & (dt > 1)
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.05, 1.3]})
    ax[0].imshow(np.ma.masked_where(~rough, dt), cmap="magma")
    my, mx = [np.array(v) for v in zip(*[c for c in ndi.center_of_mass(real, *ndi.label(real)[:1], range(1, ndi.label(real)[1] + 1))])]
    ax[0].plot(mx, my, "o", color="cyan", ms=13, mec="k", mew=1.5, label="markers (hill tops)")
    ax[0].set_title("(a) one marker per hill top = one\ncandidate grain (cyan)")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    x = np.linspace(0, 10, 400)
    prof = 7 * np.exp(-((x - 2.6) / 1.3) ** 2) + 7 * np.exp(-((x - 7.4) / 1.3) ** 2) + 0.35 * np.exp(-((x - 4.95) / 0.12) ** 2) + 1.2
    ax[1].plot(x, prof, "k", lw=2)
    ax[1].fill_between(x, 0, prof, color="#e8dcc0")
    valley = prof[(x > 4.3) & (x < 4.8)].min()
    ax[1].annotate("", xy=(2.6, prof.max()), xytext=(2.6, valley), arrowprops=dict(arrowstyle="<->", color="green", lw=2))
    ax[1].text(2.75, (prof.max() + valley) / 2, "rises far more than\nh = 0.5 px\n-> MARKER", color="green", fontsize=12)
    ax[1].annotate("tiny bump: rises LESS\nthan h = 0.5 px -> ignored", xy=(4.95, valley + 0.4), xytext=(5.6, 9.0),
                   color="firebrick", fontsize=12, arrowprops=dict(arrowstyle="->", color="firebrick"))
    ax[1].set_ylim(0, 10)
    ax[1].set_xticks([])
    ax[1].set_xlabel("position along the grains")
    ax[1].set_ylabel("distance to edge")
    ax[1].set_title("(b) the h rule: a hill becomes a marker only if\nit rises at least h above the valley next to it")
    save(fig, "step2_markers.png")


def step3():
    """Growth from the markers, shown directly on the grains: thick parts are reached first, the neck last."""
    dt = ndi.distance_transform_edt(TWO)
    markers, _ = ndi.label(morphology.h_maxima(ndi.gaussian_filter(dt, 1), 0.5) & TWO)
    lab = segmentation.watershed(-dt, markers, mask=TWO)
    my, mx = zip(*ndi.center_of_mass(markers > 0, markers, [1, 2]))
    top = dt.max()
    frames = [(top - 4, "(a) growth starts at\neach marker"),
              (top * 0.55, "(b) each grain grows outwards,\nthick parts first"),
              (dt[cuts(lab)].max() + 0.5, "(c) the two colors are about\nto meet at the thin neck"),
              (0, "(d) they meet: the meeting\nline is the cut (red)")]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
    for a_, (level, title) in zip(ax, frames):
        rgb = np.ones(TWO.shape + (3,))
        rgb[TWO] = (0.92, 0.92, 0.92)                        # grain area not reached yet: light grey
        reached = TWO & (dt >= level)
        rgb[reached & (lab == 1)] = (0.35, 0.6, 0.9)
        rgb[reached & (lab == 2)] = (1.0, 0.65, 0.3)
        if level == 0:
            rgb[ndi.binary_dilation(cuts(lab))] = (0.9, 0, 0)
        a_.imshow(rgb)
        a_.contour(TWO, [0.5], colors="0.4", linewidths=1)
        a_.plot(mx, my, "o", color="cyan", ms=11, mec="k")
        a_.set_title(title)
        a_.set_xticks([])
        a_.set_yticks([])
    fig.subplots_adjust(bottom=0.2)
    fig.text(0.5, 0.06, "Each grain grows from its marker (cyan). Because growth follows the distance map, the thick "
             "centres are filled first and the thin neck last.\nThe two colors therefore meet at the neck, and that "
             "meeting line becomes the boundary between the two grains. Grey = not reached yet.",
             ha="center", fontsize=12)
    save(fig, "step3_watershed.png")


def step4():
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
    for a, (mask, title, col) in zip(ax, [(TWO, "CASE 1 - two grains touching", "tab:green"),
                                          (WAIST, "CASE 2 - one long grain with a waist", "darkorange")]):
        dt = ndi.distance_transform_edt(mask)
        markers, _ = ndi.label(morphology.h_maxima(ndi.gaussian_filter(dt, 1), 0.5) & mask)
        lab = segmentation.watershed(-dt, markers, mask=mask)
        cut = cuts(lab)
        neck = 2 * dt[cut].max()
        width = 2 * min(ndi.maximum(dt, lab, [1, 2]))
        ratio = neck / width
        rgb = np.ones(mask.shape + (3,))
        rgb[mask] = (0.93, 0.87, 0.72)
        rgb[ndi.binary_dilation(cut)] = (0.9, 0, 0) if ratio < 0.7 else (1.0, 0.8, 0.0)
        a.imshow(rgb)
        cx = int(np.mean(np.nonzero(cut)[1]))
        ys = np.nonzero(mask[:, cx])[0]
        a.annotate("", xy=(cx + 3, ys.min()), xytext=(cx + 3, ys.max()), arrowprops=dict(arrowstyle="<->", lw=2, color="k"))
        a.text(cx + 5, CY + 3, f"neck\n{neck:.0f} px", fontsize=12)
        piece_x = int(ndi.maximum_position(dt, lab, 1)[1])
        ys2 = np.nonzero(mask[:, piece_x])[0]
        a.annotate("", xy=(piece_x, ys2.min()), xytext=(piece_x, ys2.max()), arrowprops=dict(arrowstyle="<->", lw=2, color="navy"))
        a.text(piece_x - 22, CY + 3, f"grain\n{width:.0f} px", fontsize=12, color="navy")
        verdict = "below 0.7: CUT KEPT (2 grains)" if ratio < 0.7 else "0.7 or more: CUT UNDONE (1 grain)"
        a.set_title(f"{title}\nneck ratio = {neck:.0f} / {width:.0f} = {ratio:.2f}\n{verdict}", color=col)
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("neck ratio = width of the neck / width of the smaller piece", fontsize=15, y=1.03)
    save(fig, "step4_neck_ratio.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    step1()
    step2()
    step3()
    step4()
