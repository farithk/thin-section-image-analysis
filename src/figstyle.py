"""Larger text in all figures (for reading them in the notebooks and in the printed report).

enlarge_text(fig): enlarges every text in a figure (titles, labels, ticks, legends, notes).
    - the factor grows with the figure width (wide multi-panel figures shrink most when printed);
    - titles that become wider than their panel are shrunk back until they fit;
    - radial tick labels of polar plots are left unchanged (they crowd the centre).
install(): makes plt.show() enlarge the text of every open figure before displaying it (used in the notebooks)
"""
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes
from matplotlib.text import Text
from matplotlib.ticker import NullFormatter

FACTOR = 1.7


def _auto_factor(fig, factor):
    w = fig.get_size_inches()[0]
    return max(1.15, min(factor, factor * w / 16.0))


def _fit(text, max_width_px, renderer, min_size=7.0):
    if not text.get_text():
        return
    while text.get_fontsize() > min_size and text.get_window_extent(renderer).width > max_width_px:
        text.set_fontsize(text.get_fontsize() * 0.92)


def enlarge_text(fig, factor=FACTOR, relayout=True):
    if getattr(fig, "_text_enlarged", False):
        return fig
    f = _auto_factor(fig, factor)
    skip = set()
    for ax in fig.get_axes():
        for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
            if scale == "log":                      # crowded minor labels on log axes (e.g. 6x10^1)
                axis.set_minor_formatter(NullFormatter())
        if isinstance(ax, PolarAxes):
            skip.update(id(t) for t in ax.get_yticklabels())
    for t in fig.findobj(Text):
        if id(t) not in skip:
            t.set_fontsize(t.get_fontsize() * f)
    fig._text_enlarged = True
    if relayout and not fig.get_constrained_layout():
        try:
            fig.tight_layout()
        except Exception:
            pass
    # shrink titles that no longer fit their panel
    try:
        renderer = fig.canvas.get_renderer()
        for ax in fig.get_axes():
            width = ax.get_window_extent(renderer).width
            for t in (ax.title, ax._left_title, ax._right_title):
                _fit(t, 1.08 * width, renderer)
        if fig._suptitle is not None:
            _fit(fig._suptitle, fig.bbox.width, renderer)
        if relayout and not fig.get_constrained_layout():
            fig.tight_layout()
    except Exception:
        pass
    return fig


def install(factor=FACTOR):
    if getattr(plt.show, "_enlarging", False):
        return
    _show = plt.show

    def show(*args, **kwargs):
        for n in plt.get_fignums():
            enlarge_text(plt.figure(n), factor)
        return _show(*args, **kwargs)

    show._enlarging = True
    plt.show = show
