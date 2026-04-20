"""Shared helper: build a tabular body snippet for one metric.

The body snippet is the block inside (and including) the adjustbox — no float,
no caption, no label. It can be written as a body-only file or wrapped by a
caller to produce a standalone table.

Public API
----------
build_body(agg, metric, datasets, rows, lower_is_better) -> str
"""
from __future__ import annotations
import numpy as np

# ── Dataset display names ──────────────────────────────────────────────────
DATASET_DISPLAY: dict[str, str] = {
    "CIFAR":  "CIFAR100",
    "IN-R":   "IN-R",
    "IN-A":   "IN-A",
    "CUB":    "CUB",
    "OB":     "OB",
    "VTAB":   "VTAB",
    "CARS":   "CARS",
}

# ── Canonical method display names ────────────────────────────────────────
METHOD_DISPLAY: dict[str, str] = {
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
    # KCEA variants
    "KCEA-FT":             r"KCEA$_{\text{FT}}$",
    "KCEA-FT+Merge":       r"KCEA$_{\text{FT + Merge}}$",
    "KCEA-FT+Merge+Align": r"KCEA$_{\text{FT + Merge + Align}}$",
}

# ── Baseline rows in alphabetical-by-display-name order ──────────────────
BASELINE_KEYS = [
    "APER+Adapter",
    "APER+SSF",
    "APER+VPT-deep",
    "CodaPrompt",
    "DUCT",
    "DualPrompt",
    "EASE",
    "L2P",
    "MOS",
    "SLCA",
    "TUNA",
]

KCEA_KEYS = [
    "KCEA-FT",
    "KCEA-FT+Merge",
    "KCEA-FT+Merge+Align",
]

# All rows in display order (baselines then KCEA)
ALL_KEYS = BASELINE_KEYS + KCEA_KEYS

# Default column order
DATASETS_ORDER = ["CIFAR", "IN-R", "IN-A", "CUB", "OB", "VTAB", "CARS"]


# ── Cell formatting ────────────────────────────────────────────────────────

def _cell(mean: float, std: float, bold: bool, italic: bool) -> str:
    """Format a data cell as  mean$_{\pm std}$  with optional bold/italic."""
    mean_str = f"{mean:.2f}"
    std_str  = f"{std:.2f}"
    if bold:
        mean_str = rf"\textbf{{{mean_str}}}"
    elif italic:
        mean_str = rf"\textit{{{mean_str}}}"
    return rf"{mean_str}$_{{\pm {std_str}}}$"


def _avg_cell(mean: float, bold: bool, italic: bool) -> str:
    """Format the Avg. column (mean only, no std subscript)."""
    s = f"{mean:.2f}"
    if bold:
        s = rf"\textbf{{{s}}}"
    elif italic:
        s = rf"\textit{{{s}}}"
    return s


# ── Ranking ────────────────────────────────────────────────────────────────

def _rank(values: list[float | None], lower_is_better: bool) -> tuple[int, int]:
    """Return (best_idx, second_idx) ignoring None entries."""
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        return -1, -1
    valid.sort(key=lambda x: x[1], reverse=not lower_is_better)
    best   = valid[0][0]
    second = valid[1][0] if len(valid) > 1 else -1
    return best, second


# ── Main builder ───────────────────────────────────────────────────────────

def build_body(
    agg: dict,
    metric: str,
    datasets: list[str] = DATASETS_ORDER,
    lower_is_better: bool = False,
) -> str:
    """Return the body snippet: adjustbox wrapping the tabular block.

    Parameters
    ----------
    agg             : output of loader.aggregate_all
    metric          : 'FA' | 'AA' | 'FF'
    datasets        : dataset keys in column order
    lower_is_better : True for FF
    """
    all_keys = ALL_KEYS
    n_ds = len(datasets)

    # ── Collect per-column means for ranking ──────────────────────────────
    col_means: dict[str, list[float | None]] = {ds: [] for ds in datasets}
    avg_means: list[float | None] = []

    for key in all_keys:
        row_means = []
        for ds in datasets:
            entry = agg.get(key, {}).get(ds)
            v = entry[metric][0] if entry else None
            col_means[ds].append(v)
            row_means.append(v)
        valid = [v for v in row_means if v is not None]
        avg_means.append(float(np.mean(valid)) if valid else None)

    # ── Per-column rankings ────────────────────────────────────────────────
    col_ranks = {ds: _rank(col_means[ds], lower_is_better) for ds in datasets}
    avg_rank  = _rank(avg_means, lower_is_better)

    # ── Column spec ───────────────────────────────────────────────────────
    col_spec = "l" + " c" * n_ds + " c"   # method + datasets + Avg.

    # ── Header ────────────────────────────────────────────────────────────
    ds_header = " & ".join(
        rf"\textbf{{{DATASET_DISPLAY.get(ds, ds)}}}" for ds in datasets
    )

    lines: list[str] = [
        r"\renewcommand{\arraystretch}{0.95}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\footnotesize",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Method & {ds_header} & \textbf{{Avg.}} \\",
        r"\midrule",
    ]

    def _row(key: str, global_idx: int) -> str:
        display = METHOD_DISPLAY.get(key, key)
        cells = [display]
        row_means_for_avg = []

        for j, ds in enumerate(datasets):
            entry = agg.get(key, {}).get(ds)
            if entry is None:
                cells.append("--")
                row_means_for_avg.append(None)
            else:
                mean, std = entry[metric]
                bi, si = col_ranks[ds]
                cells.append(_cell(mean, std,
                                   bold=(global_idx == bi),
                                   italic=(global_idx == si)))
                row_means_for_avg.append(mean)

        valid = [v for v in row_means_for_avg if v is not None]
        if valid:
            avg_m = float(np.mean(valid))
            bi_a, si_a = avg_rank
            cells.append(_avg_cell(avg_m,
                                   bold=(global_idx == bi_a),
                                   italic=(global_idx == si_a)))
        else:
            cells.append("--")

        return " & ".join(cells) + r" \\"

    # Baseline rows
    for i, key in enumerate(BASELINE_KEYS):
        lines.append(_row(key, i))

    # Separator + KCEA rows
    lines.append(r"\midrule")
    for ki, key in enumerate(KCEA_KEYS):
        lines.append(_row(key, len(BASELINE_KEYS) + ki))

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
    ]
    return "\n".join(lines)
