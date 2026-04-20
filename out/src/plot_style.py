"""Consistent matplotlib style for Neurocomputing figures.

Usage
-----
    from src.plot_style import apply_style, METHOD_STYLE, FIG_WIDTH
    apply_style()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH["single"], 3))
    ...
    save_fig(fig, "out/figures/my_fig.pdf")
"""

from __future__ import annotations
import matplotlib
import matplotlib.pyplot as plt

# ── Figure width constants (inches) ──────────────────────────────────────────
FIG_WIDTH = {
    "single": 3.4,
    "double": 7.0,
}
MAX_HEIGHT = 8.5  # inches

# ── Method style map ─────────────────────────────────────────────────────────
# Each entry: color, marker, linestyle, linewidth, zorder
# KCEA variants are visually distinct (orange/red family, thicker lines).
_BASE_LW = 1.0
_KCEA_LW = 2.0

METHOD_STYLE: dict[str, dict] = {
    "L2P":            {"color": "#4C72B0", "marker": "s",  "ls": "--",  "lw": _BASE_LW, "zorder": 2},
    "DualPrompt":     {"color": "#55A868", "marker": "^",  "ls": "--",  "lw": _BASE_LW, "zorder": 2},
    "CodaPrompt":     {"color": "#C44E52", "marker": "D",  "ls": "--",  "lw": _BASE_LW, "zorder": 2},
    "APER+Adapter":   {"color": "#8172B2", "marker": "v",  "ls": ":",   "lw": _BASE_LW, "zorder": 2},
    "APER+SSF":       {"color": "#937860", "marker": "P",  "ls": ":",   "lw": _BASE_LW, "zorder": 2},
    "APER+VPT":       {"color": "#DA8BC3", "marker": "X",  "ls": ":",   "lw": _BASE_LW, "zorder": 2},
    "EASE":           {"color": "#8C8C8C", "marker": "h",  "ls": "-.",  "lw": _BASE_LW, "zorder": 2},
    "SLCA":           {"color": "#64B5CD", "marker": "p",  "ls": "-.",  "lw": _BASE_LW, "zorder": 2},
    "DUCT":           {"color": "#AECDE8", "marker": "<",  "ls": "--",  "lw": _BASE_LW, "zorder": 2},
    "MOS":            {"color": "#3B7D3F", "marker": "*",  "ls": "-",   "lw": _BASE_LW + 0.3, "zorder": 3},
    "TUNA":           {"color": "#1A237E", "marker": "H",  "ls": "-",   "lw": _BASE_LW + 0.3, "zorder": 3},
    # KCEA variants — warm coral/crimson family, same marker, distinguished by shade + linestyle
    "KCEA":                  {"color": "#D84315", "marker": "o", "ls": "-",  "lw": _KCEA_LW,       "zorder": 4},
    "KCEA-FT":               {"color": "#FFAB40", "marker": "o", "ls": "--", "lw": _KCEA_LW,       "zorder": 4},
    "KCEA-FT+Merge":         {"color": "#EF5350", "marker": "o", "ls": "-.", "lw": _KCEA_LW,       "zorder": 4},
    "KCEA-FT+Merge+Align":   {"color": "#B71C1C", "marker": "o", "ls": "-",  "lw": _KCEA_LW + 0.5, "zorder": 5},
}

# Fallback style for methods not in the map
_FALLBACK_STYLE = {"color": "#999999", "marker": ".", "ls": "-", "lw": _BASE_LW, "zorder": 1}


def get_style(method: str) -> dict:
    return METHOD_STYLE.get(method, _FALLBACK_STYLE)


def apply_style() -> None:
    """Apply global matplotlib rcParams for Neurocomputing."""
    matplotlib.rcParams.update({
        "font.family":        "serif",
        "font.size":          9,
        "axes.labelsize":     9,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    8,
        "axes.grid":          True,
        "grid.linestyle":     "--",
        "grid.alpha":         0.3,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "pdf.fonttype":       42,   # embed fonts
        "ps.fonttype":        42,
    })


def save_fig(fig: plt.Figure, path: str, **kwargs) -> None:
    """Save figure as PDF with tight layout."""
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.02, **kwargs)
    print(f"  [saved] {path}")
