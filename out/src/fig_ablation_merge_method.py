"""Generate accuracy-trajectory figure for the merge_method ablation.

Output (in assets/figures/):
    fig_ablation_merge_method_vtab.pdf

Run from KCEA/:
    python -m out.src.fig_ablation_merge_method \\
        [--input ablation.json] [--out-dir out/assets/figures]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from out.src.ablation_loader import load_ablation, aggregate_ablation
from out.src.plot_style import apply_style, save_fig, FIG_WIDTH

_HERE = Path(__file__).parent
DEFAULT_INPUT   = _HERE.parent.parent / "ablation.json"
DEFAULT_OUT_DIR = _HERE.parent / "assets" / "figures"

ABLATION_NAME = "merge_method"

SETTING_STYLE = {
    "ties":    {"color": "#E65100", "ls": "-",   "lw": 2.0, "marker": "o", "label": "TIES"},
    "max":     {"color": "#4C72B0", "ls": "--",  "lw": 1.5, "marker": "s", "label": "Max"},
    "max_abs": {"color": "#55A868", "ls": "-.",  "lw": 1.5, "marker": "^", "label": "Max-Abs"},
    "min":     {"color": "#C44E52", "ls": ":",   "lw": 1.5, "marker": "D", "label": "Min"},
    "avg":     {"color": "#8172B2", "ls": "--",  "lw": 1.5, "marker": "P", "label": "Avg"},
}
SETTING_ORDER = ["ties", "max_abs", "min", "avg", "max"]


def _trajectory_stats(trajectories: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.array(trajectories)
    return arr.mean(axis=0), arr.std(axis=0)


def make_figure(agg: dict[str, dict]) -> plt.Figure:
    apply_style()
    import matplotlib
    matplotlib.rcParams.update({
        "font.size":       7,
        "axes.labelsize":  7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.titlesize":  7,
    })
    fig, ax = plt.subplots(figsize=(2.2, 2.4))

    for sv in SETTING_ORDER:
        entry = agg.get(sv)
        if entry is None:
            continue
        mean, std = _trajectory_stats(entry["trajectories"])
        n_tasks = len(mean)
        x = np.arange(1, n_tasks + 1)

        style = SETTING_STYLE.get(sv, {})
        ax.plot(
            x, mean,
            color=style.get("color", "gray"),
            ls=style.get("ls", "-"),
            lw=style.get("lw", 1.5),
            marker=style.get("marker", "o"),
            markersize=3,
            label=style.get("label", sv),
            zorder=3,
        )
        ax.fill_between(
            x, mean - std, mean + std,
            color=style.get("color", "gray"),
            alpha=0.15,
            zorder=2,
        )

    n_tasks = len(next(
        (agg[sv]["trajectories"][0] for sv in SETTING_ORDER if agg.get(sv)),
        [5] * 5,
    ))
    ax.set_xlabel("Number of tasks seen")
    ax.set_ylabel("Accumulated Accuracy AA (\\%)")
    ax.set_title("VTAB")
    ax.set_xticks(np.arange(1, n_tasks + 1))
    ax.legend(loc="lower left", framealpha=0.9, markerscale=0.6)
    fig.tight_layout()
    return fig


def main(input_path: Path = DEFAULT_INPUT, out_dir: Path = DEFAULT_OUT_DIR) -> None:
    registry = load_ablation(input_path)
    if ABLATION_NAME not in registry:
        raise KeyError(
            f"'{ABLATION_NAME}' not found in {input_path}. "
            f"Run ablation_parser.py first."
        )
    entry = registry[ABLATION_NAME]
    agg = aggregate_ablation(entry)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = make_figure(agg)
    out_path = out_dir / "fig_ablation_merge_method_vtab.pdf"
    save_fig(fig, str(out_path))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default=DEFAULT_INPUT,   type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    args = parser.parse_args()
    main(args.input, args.out_dir)
