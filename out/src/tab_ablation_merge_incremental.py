"""Generate LaTeX table for the merge_incremental ablation.

Output (in assets/tables/):
  tab_ablation_merge_incremental.tex        — standalone table float
  tab_ablation_merge_incremental_body.tex   — body-only snippet

Run from KCEA/:
    python -m out.src.tab_ablation_merge_incremental \\
        [--input ablation.json] [--out-dir out/assets/tables]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from out.src.ablation_loader import load_ablation, aggregate_ablation

_HERE = Path(__file__).parent
DEFAULT_INPUT   = _HERE.parent.parent / "ablation.json"
DEFAULT_OUT_DIR = _HERE.parent / "assets" / "tables"

ABLATION_NAME = "merge_incremental"

# Display name for each setting value
SETTING_DISPLAY = {
    "True":  r"Incremental Merge",
    "False": r"Full Merge",
}
# Ordered list of settings in table rows
SETTING_ORDER = ["True", "False"]

CAPTION = (
    r"Ablation on incremental vs.\ full re-merge on VTAB. "
    r"Mean $\pm$ std across three seeds (1993, 1994, 1995). "
    r"Best result per column in \textbf{bold}."
)
LABEL = "tab:ablation-merge-incremental"


def _cell(mean: float, std: float, bold: bool) -> str:
    s = f"{mean:.2f}"
    if bold:
        s = rf"\textbf{{{s}}}"
    return rf"{s}$_{{\pm {std:.2f}}}$"


def build_body(agg: dict[str, dict]) -> str:
    """Build the tabular body (adjustbox-wrapped)."""
    # Determine best per column
    fa_vals = [agg[sv]["FA"][0] if agg.get(sv) else None for sv in SETTING_ORDER]
    aa_vals = [agg[sv]["AA"][0] if agg.get(sv) else None for sv in SETTING_ORDER]
    ff_vals = [agg[sv]["FF"][0] if agg.get(sv) else None for sv in SETTING_ORDER]

    def best_idx(vals, lower=False):
        valid = [(i, v) for i, v in enumerate(vals) if v is not None]
        if not valid:
            return -1
        valid.sort(key=lambda x: x[1], reverse=not lower)
        return valid[0][0]

    fa_best = best_idx(fa_vals)
    aa_best = best_idx(aa_vals)
    ff_best = best_idx(ff_vals, lower=True)

    lines = [
        r"\renewcommand{\arraystretch}{0.95}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\footnotesize",
        r"\begin{tabular}{l c c c}",
        r"\toprule",
        r"Setting & \textbf{FA (\%)} $\uparrow$ & \textbf{AA (\%)} $\uparrow$ & \textbf{FF (\%)} $\downarrow$ \\",
        r"\midrule",
    ]

    for i, sv in enumerate(SETTING_ORDER):
        entry = agg.get(sv)
        if entry is None:
            lines.append(rf"{SETTING_DISPLAY.get(sv, sv)} & -- & -- & -- \\")
            continue
        fa_m, fa_s = entry["FA"]
        aa_m, aa_s = entry["AA"]
        ff_m, ff_s = entry["FF"]
        row = [
            SETTING_DISPLAY.get(sv, sv),
            _cell(fa_m, fa_s, bold=(i == fa_best)),
            _cell(aa_m, aa_s, bold=(i == aa_best)),
            _cell(ff_m, ff_s, bold=(i == ff_best)),
        ]
        lines.append(" & ".join(row) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
    ]
    return "\n".join(lines)


def _standalone(body: str, caption: str, label: str) -> str:
    return "\n".join([
        r"\begin{table}[!t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        body,
        r"\end{table}",
    ])


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

    body = build_body(agg)

    body_path = out_dir / "tab_ablation_merge_incremental_body.tex"
    body_path.write_text(body)
    print(f"  [saved] {body_path}")

    standalone_path = out_dir / "tab_ablation_merge_incremental.tex"
    standalone_path.write_text(_standalone(body, CAPTION, LABEL))
    print(f"  [saved] {standalone_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default=DEFAULT_INPUT,   type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    args = parser.parse_args()
    main(args.input, args.out_dir)
