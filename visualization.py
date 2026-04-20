"""KCEA result analysis and LaTeX table generation.

Usage:
    python visualization.py            # generates tables/main_results.tex

Metrics (see paper §Evaluation Metrics):
    FA  — Final Accuracy:       mean of the last row of the accuracy matrix
    AA  — Accumulated Accuracy: mean of per-task FA_t (i.e., mean of all row means)
    FF  — Final Forgetting:     mean of (peak_accuracy_on_task_i - final_accuracy_on_task_i)
                                over all tasks except the last
"""

import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent

# ── Ordering ──────────────────────────────────────────────────────────────────
DATASETS = ["CIFAR", "CARS", "CUB", "IN-A", "IN-R", "OB", "VTAB"]

METHODS = [
    "L2P",
    "DualPrompt",
    "CodaPrompt",
    "APER+Adapter",
    "APER+SSF",
    "APER+VPT-deep",
    "EASE",
    "SLCA",
    "DUCT",
    "MOS",
    "TUNA",
]

# Display names in the LaTeX table
METHOD_DISPLAY = {
    "L2P":          "L2P",
    "DualPrompt":   "DualPrompt",
    "CodaPrompt":   "CodaPrompt",
    "APER+Adapter": "APER+Adapter",
    "APER+SSF":     "APER+SSF",
    "APER+VPT-deep":"APER+VPT",
    "EASE":         "EASE",
    "SLCA":         "SLCA",
    "DUCT":         "DUCT",
    "MOS":          "MOS",
    "TUNA":         "TUNA",
}

EXPECTED_SEEDS = 3


# ── Metric computation ────────────────────────────────────────────────────────

def compute_fa(perf: list) -> float:
    """FA = (1/T) * sum_i A_{T,i}  — mean of the last row."""
    return float(np.mean(perf[-1]))


def compute_aa(perf: list) -> float:
    """AA = (1/T) * sum_t FA_t  — mean of every row mean."""
    return float(np.mean([np.mean(row) for row in perf]))


def compute_ff(perf: list) -> float:
    """FF = (1/(T-1)) * sum_{i=1}^{T-1} (max_{t>=i} A_{t,i} - A_{T,i}).

    The last task is excluded because it has no later stage to forget at.
    """
    T = len(perf)
    if T < 2:
        return 0.0
    forgetting = []
    for i in range(T - 1):          # task index 0 .. T-2
        col = [perf[t][i] for t in range(i, T)]
        forgetting.append(max(col) - perf[-1][i])
    return float(np.mean(forgetting))


# ── Load & aggregate ──────────────────────────────────────────────────────────

def load_results() -> dict:
    path = ROOT / "result.json"
    with open(path) as f:
        return json.load(f)


def aggregate(results: dict) -> dict:
    """Compute per-seed metrics and return mean/std over seeds.

    Returns
    -------
    agg : dict  {method -> {dataset -> {"FA": (mean, std), "AA": ..., "FF": ...,
                                        "n_seeds": int} | None}}
    """
    agg = {}
    for method in METHODS:
        agg[method] = {}
        method_data = results.get(method, {})
        for dataset in DATASETS:
            seeds_data = method_data.get(dataset, {})
            fa_vals, aa_vals, ff_vals = [], [], []
            for entry in seeds_data.values():
                perf = entry.get("performance", [])
                if not perf:
                    continue
                fa_vals.append(compute_fa(perf))
                aa_vals.append(compute_aa(perf))
                ff_vals.append(compute_ff(perf))

            if not fa_vals:
                agg[method][dataset] = None
                continue

            agg[method][dataset] = {
                "FA": (float(np.mean(fa_vals)), float(np.std(fa_vals))),
                "AA": (float(np.mean(aa_vals)), float(np.std(aa_vals))),
                "FF": (float(np.mean(ff_vals)), float(np.std(ff_vals))),
                "n_seeds": len(fa_vals),
            }
    return agg


# ── LaTeX helpers ─────────────────────────────────────────────────────────────

def _fmt(mean: float, std: float, bold: bool = False, underline: bool = False,
         incomplete: bool = False) -> str:
    s = rf"{mean:.2f}{{\tiny{{$\pm${std:.2f}}}}}"
    if incomplete:
        s += r"$^\dagger$"
    if bold:
        s = rf"\textbf{{{s}}}"
    if underline:
        s = rf"\underline{{{s}}}"
    return s


def _rank(col_means: list, lower_is_better: bool) -> tuple:
    """Return (best_row_idx, second_best_row_idx) among non-None values."""
    valid = [(i, v) for i, v in enumerate(col_means) if v is not None]
    if not valid:
        return -1, -1
    valid.sort(key=lambda x: x[1], reverse=not lower_is_better)
    best = valid[0][0]
    second = valid[1][0] if len(valid) > 1 else -1
    return best, second


# ── Table generation ──────────────────────────────────────────────────────────

def make_latex_table(agg: dict, datasets: list) -> str:
    """Build one LaTeX tabular block for the given dataset subset.

    Columns: Method | (FA  AA  FF) × len(datasets)
    Best per column in bold, second-best underlined.
    Entries based on fewer than EXPECTED_SEEDS seeds are marked with †.
    """
    METRICS = ["FA", "AA", "FF"]
    LOWER_BETTER = {"FA": False, "AA": False, "FF": True}

    n_ds = len(datasets)

    # Collect column values for ranking
    col_means = {ds: {m: [] for m in METRICS} for ds in datasets}
    for method in METHODS:
        for ds in datasets:
            entry = agg[method].get(ds)
            for metric in METRICS:
                col_means[ds][metric].append(entry[metric][0] if entry else None)

    # Determine best / second-best per (dataset, metric) column
    ranks = {}
    for ds in datasets:
        ranks[ds] = {}
        for metric in METRICS:
            ranks[ds][metric] = _rank(col_means[ds][metric], LOWER_BETTER[metric])

    # Build lines
    lines = []

    col_spec = "l" + (" ccc" * n_ds)
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Row 1: dataset name spanning 3 columns
    header1 = "Method"
    for ds in datasets:
        header1 += rf" & \multicolumn{{3}}{{c}}{{\textbf{{{ds}}}}}"
    lines.append(header1 + r" \\")

    # Cmidrules
    cmid = ""
    for k in range(n_ds):
        start = 2 + k * 3
        cmid += rf"\cmidrule(lr){{{start}-{start + 2}}}"
    lines.append(cmid)

    # Row 2: FA  AA  FF repeated
    header2 = ""
    for _ in datasets:
        header2 += r" & FA & AA & FF"
    lines.append(header2 + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for mi, method in enumerate(METHODS):
        row = METHOD_DISPLAY[method]
        for ds in datasets:
            entry = agg[method].get(ds)
            for metric in METRICS:
                if entry is None:
                    row += " & ---"
                else:
                    mean, std = entry[metric]
                    incomplete = entry["n_seeds"] < EXPECTED_SEEDS
                    bi, si = ranks[ds][metric]
                    row += " & " + _fmt(
                        mean, std,
                        bold=(mi == bi),
                        underline=(mi == si),
                        incomplete=incomplete,
                    )
        lines.append(row + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    return "\n".join(lines)


def make_full_table(agg: dict) -> str:
    """Wrap two tabular halves in a single sidewaystable float."""
    # Split 7 datasets into two rows to keep width manageable
    half1 = DATASETS[:4]   # CIFAR  CARS  CUB  IN-A
    half2 = DATASETS[4:]   # IN-R   OB    VTAB

    incomplete_methods = set()
    for method in METHODS:
        for ds in DATASETS:
            entry = agg[method].get(ds)
            if entry and entry["n_seeds"] < EXPECTED_SEEDS:
                incomplete_methods.add(METHOD_DISPLAY[method])

    dagger_note = ""
    if incomplete_methods:
        names = ", ".join(sorted(incomplete_methods))
        dagger_note = (
            rf"$^\dagger$Fewer than {EXPECTED_SEEDS} seeds available "
            rf"({names}); mean and std computed over available seeds."
        )

    caption = (
        r"Comparison of continual learning baselines across seven benchmarks. "
        r"\textbf{FA}: Final Accuracy (\%), "
        r"\textbf{AA}: Accumulated Accuracy (\%), "
        r"\textbf{FF}: Final Forgetting (\%); "
        r"all values are mean\,$\pm$\,std over three seeds. "
        r"\textbf{Bold}: best result per column; \underline{underlined}: second best. "
        + dagger_note
    )

    lines = []
    lines.append(r"\begin{sidewaystable}")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\small")

    lines.append(make_latex_table(agg, half1))
    lines.append(r"\vspace{4pt}")
    lines.append(make_latex_table(agg, half2))

    lines.append(r"\end{sidewaystable}")

    return "\n".join(lines)


# ── Console summary ───────────────────────────────────────────────────────────

def print_summary(agg: dict) -> None:
    """Print a plain-text summary table to stdout."""
    col_w = 14

    for metric in ["FA", "AA", "FF"]:
        print(f"\n{'─' * (16 + col_w * len(DATASETS))}")
        print(f"  {metric}")
        print(f"{'─' * (16 + col_w * len(DATASETS))}")
        header = f"{'Method':<16}" + "".join(f"{ds:>{col_w}}" for ds in DATASETS)
        print(header)
        for method in METHODS:
            row = f"{METHOD_DISPLAY[method]:<16}"
            for ds in DATASETS:
                entry = agg[method].get(ds)
                if entry is None:
                    row += f"{'---':>{col_w}}"
                else:
                    mean, std = entry[metric]
                    tag = "*" if entry["n_seeds"] < EXPECTED_SEEDS else ""
                    row += f"{mean:>7.2f}±{std:.2f}{tag}".rjust(col_w)
            print(row)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    results = load_results()
    agg = aggregate(results)

    print_summary(agg)

    out_dir = ROOT / "tables"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "main_results.tex"
    out_path.write_text(make_full_table(agg))
    print(f"\nLaTeX table written to {out_path}")


if __name__ == "__main__":
    main()
