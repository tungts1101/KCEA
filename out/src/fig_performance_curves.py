"""Per-dataset AA trajectory curves for all methods in the main table.

Layout: 2 rows × 4 cols (7 dataset subplots + 1 legend-only panel).
Each subplot shows accumulated accuracy (mean of row t) at each task step,
averaged ± std across seeds.

Output (in assets/figures/):
    fig_performance_curves.pdf

Run from KCEA/:
    python -m out.src.fig_performance_curves [--input result.json] [--out-dir out/assets/figures]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from out.src.loader import load_results
from out.src.plot_style import apply_style, save_fig, get_style
from out.src.class_schedule import SCHEDULES

_HERE = Path(__file__).parent
DEFAULT_INPUT   = _HERE.parent.parent / "result.json"
DEFAULT_OUT_DIR = _HERE.parent / "assets" / "figures"

DATASETS_ORDER = ["CIFAR", "IN-R", "IN-A", "CUB", "OB", "VTAB", "CARS"]

DATASET_DISPLAY = {
    "CIFAR": "CIFAR100",
    "IN-R":  "IN-R",
    "IN-A":  "IN-A",
    "CUB":   "CUB",
    "OB":    "OB",
    "VTAB":  "VTAB",
    "CARS":  "CARS",
}

# Display name for legend
METHOD_DISPLAY = {
    "APER+Adapter":        "APER+Adapter",
    "APER+SSF":            "APER+SSF",
    "APER+VPT-deep":       "APER+VPT-Deep",
    "CodaPrompt":          "CODA-Prompt",
    "DUCT":                "DUCT",
    "DualPrompt":          "DualPrompt",
    "EASE":                "EASE",
    "L2P":                 "L2P",
    "MOS":                 "MOS",
    "SLCA":                "SLCA",
    "TUNA":                "TUNA",
    "KCEA-FT":             "KCEA (FT)",
    "KCEA-FT+Merge":       "KCEA (FT+Merge)",
    "KCEA-FT+Merge+Align": "KCEA (FT+Merge+Align)",
}

# Style key mapping (result.json key → plot_style key)
STYLE_KEY = {
    "APER+VPT-deep":       "APER+VPT",
    "CodaPrompt":          "CodaPrompt",
}

METHODS_ORDER = [
    "APER+Adapter", "APER+SSF", "APER+VPT-deep",
    "CodaPrompt", "DUCT", "DualPrompt", "EASE", "L2P",
    "MOS", "SLCA", "TUNA",
    "KCEA-FT", "KCEA-FT+Merge", "KCEA-FT+Merge+Align",
]

KCEA_METHODS = {"KCEA-FT", "KCEA-FT+Merge", "KCEA-FT+Merge+Align"}


def _group_sizes(dataset: str) -> list[int] | None:
    cum = SCHEDULES.get(dataset)
    if cum is None:
        return None
    sizes = [cum[0]] + [cum[t] - cum[t - 1] for t in range(1, len(cum))]
    return None if len(set(sizes)) == 1 else sizes


def _trajectory(perf: list[list[float]], group_sizes: list[int] | None = None) -> list[float]:
    """Per-task class-weighted accumulated accuracy = weighted mean of row t."""
    if group_sizes is None:
        return [float(np.mean(row)) for row in perf]
    return [float(np.average(row, weights=group_sizes[:len(row)])) for row in perf]


def _method_trajectories(
    raw: dict, method: str, dataset: str
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (mean, std) trajectory arrays across seeds, or (None, None)."""
    seed_dict = raw.get(method, {}).get(dataset, {})
    gs = _group_sizes(dataset)
    trajs = []
    for seed, run in seed_dict.items():
        perf = run.get("performance", [])
        if perf:
            trajs.append(_trajectory(perf, gs))
    if not trajs:
        return None, None
    arr = np.array(trajs)
    return arr.mean(axis=0), arr.std(axis=0)


_LINE_LW = 0.8   # uniform thickness for all methods


_LINE_LW = 0.8   # uniform thickness for all methods


def make_figure(raw: dict, show_std: bool = False) -> plt.Figure:
    apply_style()
    import matplotlib
    import matplotlib.gridspec as gridspec
    matplotlib.rcParams.update({
        "font.size":       7,
        "axes.labelsize":  7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.titlesize":  7,
    })

    # Layout (3, 2, 2): row 0 has 3 cols, rows 1-2 have 2 cols + shared legend
    fig = plt.figure(figsize=(7.0, 5.0))
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.55, wspace=0.35)

    # Row 0: 3 subplots
    subplot_specs = [
        gs[0, 0], gs[0, 1], gs[0, 2],   # CIFAR, IN-R, IN-A
        gs[1, 0], gs[1, 1],              # CUB, OB
        gs[2, 0], gs[2, 1],              # VTAB, CARS
    ]
    data_axes = [fig.add_subplot(spec) for spec in subplot_specs]

    # Legend spans last column of rows 1-2
    legend_ax = fig.add_subplot(gs[1:, 2])
    legend_ax.axis("off")

    for ax, ds in zip(data_axes, DATASETS_ORDER):
        ax.set_title(DATASET_DISPLAY[ds])

        for method in METHODS_ORDER:
            mean, std = _method_trajectories(raw, method, ds)
            if mean is None:
                continue

            style_key = STYLE_KEY.get(method, method)
            st = get_style(style_key)
            x = np.arange(1, len(mean) + 1)

            ax.plot(
                x, mean,
                color=st["color"], ls=st["ls"], lw=_LINE_LW,
                marker=st["marker"], markersize=2,
                zorder=st["zorder"],
            )
            if show_std:
                ax.fill_between(
                    x, mean - std, mean + std,
                    color=st["color"], alpha=0.12, zorder=st["zorder"] - 1,
                )

        ax.set_xlabel("Tasks")
        ax.set_ylabel("AA (%)")
        n_tasks = next(
            (len(next(iter(raw.get(m, {}).get(ds, {}).values()))["performance"])
             for m in METHODS_ORDER if raw.get(m, {}).get(ds)),
            10,
        )
        ax.set_xticks(np.arange(1, n_tasks + 1))

    # ── Legend (vertical, spanning last 2 rows of col 2) ─────────────────────
    handles = []
    for method in METHODS_ORDER:
        has_data = any(raw.get(method, {}).get(ds) for ds in DATASETS_ORDER)
        if not has_data:
            continue
        style_key = STYLE_KEY.get(method, method)
        st = get_style(style_key)
        handle = mlines.Line2D(
            [], [],
            color=st["color"], ls=st["ls"], lw=_LINE_LW * 1.5,
            marker=st["marker"], markersize=3,
            label=METHOD_DISPLAY.get(method, method),
        )
        handles.append(handle)

    legend_ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        ncol=1,
        fontsize=6,
        handlelength=2.2,
    )

    fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.05)
    return fig


def main(input_path: Path = DEFAULT_INPUT, out_dir: Path = DEFAULT_OUT_DIR) -> None:
    raw = load_results(input_path)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Main figure — clean lines, no shading
    fig = make_figure(raw, show_std=False)
    save_fig(fig, str(out_dir / "fig_performance_curves.pdf"))
    plt.close(fig)

    # Ablation figure — same layout with ±std shading
    fig_std = make_figure(raw, show_std=True)
    save_fig(fig_std, str(out_dir / "fig_performance_curves_std.pdf"))
    plt.close(fig_std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default=DEFAULT_INPUT,   type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    args = parser.parse_args()
    main(args.input, args.out_dir)
