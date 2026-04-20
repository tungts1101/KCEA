"""Shared LaTeX table utilities for tab_overall_*.py scripts."""

from __future__ import annotations
import warnings


# ── Canonical ordering ────────────────────────────────────────────────────────

DATASETS_ORDER = ["CIFAR", "IN-R", "IN-A", "CUB", "OB", "VTAB", "CARS"]

DATASET_DISPLAY = {
    "CIFAR":  "CIFAR100",
    "IN-R":   "IN-R",
    "IN-A":   "IN-A",
    "CUB":    "CUB",
    "OB":     "OB",
    "VTAB":   "VTAB",
    "CARS":   "CARS",
}

# Baseline methods (in display order)
BASELINE_METHODS = [
    "L2P", "DualPrompt", "CodaPrompt",
    "APER+Adapter", "APER+SSF", "APER+VPT-deep",
    "EASE", "SLCA", "DUCT", "MOS", "TUNA",
]

# KCEA variants (displayed below a \midrule separator)
KCEA_METHODS = [
    "KCEA-FT",
    "KCEA-FT+Merge",
    "KCEA-FT+Merge+Align",
]

# Display names for methods
METHOD_DISPLAY = {
    "L2P":                   "L2P",
    "DualPrompt":            "DualPrompt",
    "CodaPrompt":            "CodaPrompt",
    "APER+Adapter":          "APER+Adapter",
    "APER+SSF":              "APER+SSF",
    "APER+VPT-deep":         "APER+VPT",
    "EASE":                  "EASE",
    "SLCA":                  r"SLCA$^\dagger$",
    "DUCT":                  "DUCT",
    "MOS":                   "MOS",
    "TUNA":                  "TUNA",
    "KCEA-FT":               r"\textit{KCEA (FT only)}",
    "KCEA-FT+Merge":         r"\textit{KCEA (FT+Merge)}",
    "KCEA-FT+Merge+Align":   r"\textbf{KCEA (Full)}",
}


# ── Cell formatting ───────────────────────────────────────────────────────────

def _fmt_cell(mean: float, std: float, bold: bool = False, italic: bool = False) -> str:
    """Format mean±std as LaTeX cell content."""
    inner = rf"{mean:.2f}_{{\pm{std:.2f}}}"
    if bold:
        inner = rf"\mathbf{{{inner}}}"
    elif italic:
        inner = rf"\mathit{{{inner}}}"
    return f"${inner}$"


def _fmt_placeholder() -> str:
    return r"\multicolumn{1}{c}{--}"


# ── Ranking ───────────────────────────────────────────────────────────────────

def _rank_column(means: list[float | None], lower_is_better: bool) -> tuple[int, int]:
    """Return (best_idx, second_idx) ignoring None entries."""
    valid = [(i, v) for i, v in enumerate(means) if v is not None]
    if not valid:
        return -1, -1
    valid.sort(key=lambda x: x[1], reverse=not lower_is_better)
    best = valid[0][0]
    second = valid[1][0] if len(valid) > 1 else -1
    return best, second


# ── Main table builder ────────────────────────────────────────────────────────

def build_metric_table(
    agg: dict,
    metric: str,
    datasets: list[str],
    caption: str,
    label: str,
    lower_is_better: bool = False,
) -> str:
    """Build a standalone LaTeX table snippet for one metric.

    Parameters
    ----------
    agg         : output of loader.aggregate_all
    metric      : one of 'FA', 'AA', 'FF'
    datasets    : list of dataset keys (JSON spelling) in column order
    caption     : LaTeX caption string
    label       : LaTeX label string
    lower_is_better : for FF=True, FA/AA=False
    """
    all_methods = BASELINE_METHODS + KCEA_METHODS
    n_ds = len(datasets)

    # ── Collect means per column for ranking ──────────────────────────────────
    # Ranking is over all methods (baselines + KCEA) combined
    col_means: dict[str, list[float | None]] = {ds: [] for ds in datasets}
    avg_means: list[float | None] = []

    for method in all_methods:
        ds_means = []
        for ds in datasets:
            entry = agg.get(method, {}).get(ds)
            v = entry[metric][0] if entry else None
            col_means[ds].append(v)
            ds_means.append(v)
        valid = [v for v in ds_means if v is not None]
        avg_means.append(float(sum(valid) / len(valid)) if valid else None)

    # Determine best/second per column
    col_ranks: dict[str, tuple[int, int]] = {}
    for ds in datasets:
        col_ranks[ds] = _rank_column(col_means[ds], lower_is_better)
    avg_ranks = _rank_column(avg_means, lower_is_better)

    # ── Build LaTeX ───────────────────────────────────────────────────────────
    col_spec = "l" + " c" * n_ds + " c"   # method + n_ds datasets + Average
    ds_headers = " & ".join(rf"\textbf{{{DATASET_DISPLAY[ds]}}}" for ds in datasets)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{adjustbox}{max width=\textwidth}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Method & {ds_headers} & \textbf{{Average}} \\",
        r"\midrule",
    ]

    def _row(method: str, mi_global: int) -> str:
        cells = [METHOD_DISPLAY.get(method, method)]
        ds_means_row = []
        for j, ds in enumerate(datasets):
            entry = agg.get(method, {}).get(ds)
            if entry is None:
                cells.append(_fmt_placeholder())
                ds_means_row.append(None)
            else:
                mean, std = entry[metric]
                bi, si = col_ranks[ds]
                cells.append(_fmt_cell(mean, std,
                                       bold=(mi_global == bi),
                                       italic=(mi_global == si)))
                ds_means_row.append(mean)
        # Average column
        valid = [v for v in ds_means_row if v is not None]
        if valid:
            avg_m = sum(valid) / len(valid)
            # std of averages across datasets (cross-dataset spread, not seed std)
            import numpy as np
            avg_s_list = []
            for ds in datasets:
                entry = agg.get(method, {}).get(ds)
                if entry:
                    avg_s_list.append(entry[metric][0])
            avg_s = float(np.std(avg_s_list)) if len(avg_s_list) > 1 else 0.0
            bi_a, si_a = avg_ranks
            cells.append(_fmt_cell(avg_m, avg_s,
                                   bold=(mi_global == bi_a),
                                   italic=(mi_global == si_a)))
        else:
            cells.append(_fmt_placeholder())
        return " & ".join(cells) + r" \\"

    # Baseline rows
    for mi, method in enumerate(BASELINE_METHODS):
        lines.append(_row(method, mi))

    # Separator + KCEA rows
    lines.append(r"\midrule")
    for ki, method in enumerate(KCEA_METHODS):
        mi_global = len(BASELINE_METHODS) + ki
        lines.append(_row(method, mi_global))

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{table}",
    ]
    return "\n".join(lines)
