"""Explanatory figure for the two split parameters of notebook 02: h (markers) and the neck ratio (merge).

Rows 1-2: two real cases from the small image (a clear touching pair; box 2 of the merge overview)
   left : distance-to-edge map with a path from one piece's centre, through the widest point of the contact, to the other
   right: distance profile along that line = two hills and a dip; h and the neck ratio computed by the real code
Row 3: number of grains vs h (no merge) and vs neck ratio (h = 0.5)

Run from the project root:  python scripts/h_neck_diagram.py
Output: outputs/figures/h_neck_ratio.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import measure

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from objects import neck_table, split_objects  # noqa: E402
from segmentation import GRAIN  # noqa: E402
from split_check_diagram import biggest_segment, cuts  # noqa: E402

PHASES = os.path.join(ROOT, "outputs", "phases_small.npy")
OUT = os.path.join(ROOT, "outputs", "figures", "h_neck_ratio.png")
H, NECK = 0.5, 0.7


def pair_at(ws, yx, r=3):
    """The two watershed pieces meeting at the cut near yx."""
    y, x = int(yx[0]), int(yx[1])
    win = ws[y - r:y + r + 1, x - r:x + r + 1]
    ids, cnt = np.unique(win[win > 0], return_counts=True)
    i, j = ids[np.argsort(-cnt)[:2]]
    return int(min(i, j)), int(max(i, j))


def saddle_point(ws, dt, i, j):
    """Pixel of piece i on its boundary with piece j where the contact is widest (= the neck the code uses)."""
    best, pos = -1.0, None
    for a_, b_, da, db, off in ((ws[:, :-1], ws[:, 1:], dt[:, :-1], dt[:, 1:], (0, 0)),
                                (ws[:-1, :], ws[1:, :], dt[:-1, :], dt[1:, :], (0, 0))):
        m = ((a_ == i) & (b_ == j)) | ((a_ == j) & (b_ == i))
        if m.any():
            v = np.where(m, np.minimum(da, db), -1)
            k = np.unravel_index(np.argmax(v), v.shape)
            if v[k] > best:
                best, pos = v[k], k
    return pos


def main():
    mask = np.load(PHASES) == GRAIN
    final, st = split_objects(mask, return_steps=True)
    ws, dt = st["watershed"], st["dt"]
    kept = cuts(ws) & cuts(final)
    undone = cuts(ws) & ~cuts(final) & mask
    area = ndi.sum(np.ones_like(final), final, index=np.arange(final.max() + 1))
    spot_keep, _ = biggest_segment(kept)
    spot_undo, _ = biggest_segment(undone, undone * area[final])

    tab = neck_table(ws, dt)
    necks = tab.set_index(["i", "j"])["neck"]
    peak = ndi.maximum(dt, ws, index=np.arange(ws.max() + 1))
    # clear "keep" example: two good-sized pieces that stay separate, with a narrow neck
    fin_of = ndi.maximum(final, ws, index=np.arange(ws.max() + 1)).astype(int)
    tab["small_peak"] = np.minimum(peak[tab.i], peak[tab.j])
    tab["ratio"] = tab.neck / tab.small_peak
    ok = tab[(fin_of[tab.i] != fin_of[tab.j]) & (tab.ratio < 0.4) & (tab.small_peak >= 6)]
    best = ok.loc[ok.small_peak.idxmax()]
    keep_pair = (int(best.i), int(best.j))

    fig = plt.figure(figsize=(18, 17))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.35], hspace=0.38, wspace=0.15)
    cases = [(keep_pair, "CASE 1 - two grains touching", "KEEP the cut: two grains", "red"),
             (pair_at(ws, spot_undo), "CASE 2 - one large grain with a 'waist' (box 2)", "UNDO the cut: one grain",
              "#b8860b")]
    for r, ((i, j), title, verdict, col) in enumerate(cases):
        pa = np.unravel_index(np.argmax(np.where(ws == i, dt, -1)), dt.shape)
        pb = np.unravel_index(np.argmax(np.where(ws == j, dt, -1)), dt.shape)
        neck, small_peak = necks.loc[(i, j)], min(peak[i], peak[j])
        ratio = neck / small_peak

        # left: distance map around the pair
        pad = 12
        y0, y1 = max(min(pa[0], pb[0]) - int(peak.max() / 2) - pad, 0), min(max(pa[0], pb[0]) + pad + 20, dt.shape[0])
        x0, x1 = max(min(pa[1], pb[1]) - pad - 20, 0), min(max(pa[1], pb[1]) + pad + 20, dt.shape[1])
        a = fig.add_subplot(gs[r, 0])
        a.imshow(np.ma.masked_where(~mask[y0:y1, x0:x1], dt[y0:y1, x0:x1]), cmap="magma", interpolation="nearest")
        cut = cuts(ws)[y0:y1, x0:x1] & ((ws[y0:y1, x0:x1] == i) | (ws[y0:y1, x0:x1] == j))
        cy, cx = np.nonzero(cut)
        a.plot(cx, cy, "s", color=col, ms=3)
        sp = saddle_point(ws, dt, i, j)
        a.plot([pa[1] - x0, sp[1] - x0, pb[1] - x0], [pa[0] - y0, sp[0] - y0, pb[0] - y0], "c-", lw=2)
        a.plot(sp[1] - x0, sp[0] - y0, "*", color="white", ms=16, mec="k")
        a.plot([pa[1] - x0, pb[1] - x0], [pa[0] - y0, pb[0] - y0], "o", color="cyan", ms=9, mec="k")
        a.text(pa[1] - x0, pa[0] - y0 - 4, "A", color="cyan", fontsize=15, fontweight="bold", ha="center")
        a.text(pb[1] - x0, pb[0] - y0 - 4, "B", color="cyan", fontsize=15, fontweight="bold", ha="center")
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(f"{title}\ndistance to edge (bright = far); cyan = path A -> contact -> B;\n"
                    f"{'red' if r == 0 else 'yellow'} squares = the watershed cut\nwhite star = widest point of the contact",
                    fontsize=11, loc="left")

        # right: profile along the line
        p1 = measure.profile_line(dt, pa, sp, order=1, mode="constant")
        p2 = measure.profile_line(dt, sp, pb, order=1, mode="constant")
        prof = np.concatenate([p1, p2[1:]])
        s = np.arange(len(prof))
        xm = len(p1) - 1
        b = fig.add_subplot(gs[r, 1])
        b.fill_between(s, 0, prof, color="#e8dcc0")
        b.plot(s, prof, "k-", lw=2)
        b.axhline(peak[i], color="0.4", ls=":", lw=1)
        b.axhline(peak[j], color="0.4", ls=":", lw=1)
        b.axhline(neck, color=col, ls="--", lw=2)
        b.text(s[-1], neck + 0.25, f"neck (contact half-width) = {neck:.1f} px", color=col, ha="right",
               fontsize=11, fontweight="bold")
        b.text(0, peak[i] + 0.25, f"peak A = {peak[i]:.1f} px", fontsize=10)
        b.text(s[-1], peak[j] + 0.25, f"peak B = {peak[j]:.1f} px", fontsize=10, ha="right")
        b.axvline(xm, color="0.5", lw=1)
        b.text(xm, 0.5, " contact", fontsize=10, color="0.3")
        b.annotate("", xy=(xm, small_peak), xytext=(xm, neck), arrowprops=dict(arrowstyle="<->", lw=2, color="tab:blue"))
        b.text(0.62, 0.3, f"blue arrow: the smaller hill rises {small_peak - neck:.1f} px above the neck\n"
                          f"-> more than h = {H} px, so each piece got its own marker",
               transform=b.transAxes, color="tab:blue", fontsize=10, ha="center",
               bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
        b.set_ylim(0, max(peak[i], peak[j]) * 1.45)
        b.set_xlabel("position along the path A -> contact -> B (px)")
        b.set_ylabel("distance to edge (px)")
        b.text(0.02, 0.97, f"neck ratio = neck / smaller peak = {neck:.1f} / {small_peak:.1f} = {ratio:.2f}\n"
                           f"{'< ' if ratio < NECK else '>= '}{NECK}  ->  {verdict}",
               transform=b.transAxes, va="top", fontsize=13, fontweight="bold", color=col,
               bbox=dict(facecolor="white", edgecolor=col, boxstyle="round,pad=0.5"))
        b.set_title("distance profile along the path: two hills (the pieces) and a dip at the contact",
                    fontsize=11, loc="left")

    # row 3: sensitivity
    hs = [0.5, 1, 1.5, 2, 3]
    n_h = [split_objects(mask, h=v, neck_ratio=None).max() for v in hs]
    rs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_r = [split_objects(mask, h=H, neck_ratio=v).max() for v in rs]
    a = fig.add_subplot(gs[2, 0])
    a.plot(hs, n_h, "o-", lw=2)
    a.axvline(H, color="tab:green", ls="--", lw=2, label=f"used: h = {H}")
    a.set_xlabel("h (px)")
    a.set_ylabel("number of grains (no merge)")
    a.legend()
    a.set_title("h controls how many markers (candidate grains) the watershed starts from:\n"
                "higher h -> fewer cuts, but real touching grains stay merged", fontsize=11, loc="left")
    b = fig.add_subplot(gs[2, 1])
    b.plot(rs, n_r, "o-", lw=2, color="tab:red")
    b.axvline(NECK, color="tab:green", ls="--", lw=2, label=f"used: neck ratio = {NECK}")
    b.set_xlabel("neck ratio threshold")
    b.set_ylabel(f"number of grains (h = {H} + merge)")
    b.legend()
    b.set_title("the neck ratio decides which cuts are undone:\n"
                "lower threshold -> more cuts undone -> fewer grains", fontsize=11, loc="left")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=85, bbox_inches="tight")
    print("saved", OUT)
    print("h:", dict(zip(hs, n_h)), " neck:", dict(zip(rs, n_r)))


if __name__ == "__main__":
    main()
