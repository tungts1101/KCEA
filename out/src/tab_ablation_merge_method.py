"""Generate LaTeX table for the merge_method ablation.

Output (in assets/tables/):
  tab_ablation_merge_method.tex        — standalone table float
  tab_ablation_merge_method_body.tex   — body-only snippet

Run from KCEA/:
    python -m out.src.tab_ablation_merge_method \\
        [--input ablation.json] [--out-dir out/assets/tables]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from out.src.ablation_loader import load_ablation, aggregate_ablation

_HERE = Path(__file__).parent
DEFAULT_INPUT   = _HERE.parent.parent / "ablation.json"
DEFAULT_OUT_DIR = _HERE.parent / "assets" / "tables"

ABLATION_NAME = "merge_method"

SETTING_DISPLAY = {
    "ties":    "TIES",
    "max":     "Max",
    "max_abs": "Max-Abs",
    "min":     "Min",
    "avg":     "Avg",
}
SETTING_ORDER = ["ties", "max", "max_abs", "min", "avg"]

CAPTION = (
    r"Ablation on task-vector merge method on VTAB. "
    r"Mean $\pm$ std across three seeds (1993, 1994, 1995). "
    r"Best result per column in \textbf{bold}, second best in \textit{italic}."
)
LABEL = "tab:ablation-merge-method"


def _cell(mean: float, std: float, bold: bool, italic: bool) -> str:
    s = f"{mean:.2f}"
    if bold:
        s = rf"\textbf{{{s}}}"
    elif italic:
        s = rf"\textit{{{s}}}"
    return rf"{s}$_{{\pm {std:.2f}}}$"


def _rank(vals: list[float | None], lower: bool = False) -> tuple[int, int]:
    valid = [(i, v) for i, v in enumerate(vals) if v is not None]
    if not valid:
        return -1, -1
    valid.sort(key=lambda x: x[1], reverse=not lower)
    best = valid[0][0]
    second = valid[1][0] if len(valid) > 1 else -1
    return best, second


def build_body(agg: dict[str, dict]) -> str:
    fa_vals = [agg[sv]["FA"][0] if agg.get(sv) else None for sv in SETTING_ORDER]
    aa_vals = [agg[sv]["AA"][0] if agg.get(sv) else None for sv in SETTING_ORDER]
    ff_vals = [agg[sv]["FF"][0] if agg.get(sv) else None for sv in SETTING_ORDER]

    fa_best, fa_2nd = _rank(fa_vals)
    aa_best, aa_2nd = _rank(aa_vals)
    ff_best, ff_2nd = _rank(ff_vals, lower=True)

    lines = [
        r"\renewcommand{\arraystretch}{0.95}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\footnotesize",
        r"\begin{tabular}{l c c c}",
        r"\toprule",
        r"Merge method & \textbf{FA (\%)} $\uparrow$ & \textbf{AA (\%)} $\uparrow$ & \textbf{FF (\%)} $\downarrow$ \\",
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
            _cell(fa_m, fa_s, bold=(i == fa_best), italic=(i == fa_2nd)),
            _cell(aa_m, aa_s, bold=(i == aa_best), italic=(i == aa_2nd)),
            _cell(ff_m, ff_s, bold=(i == ff_best), italic=(i == ff_2nd)),
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

    body_path = out_dir / "tab_ablation_merge_method_body.tex"
    body_path.write_text(body)
    print(f"  [saved] {body_path}")

    standalone_path = out_dir / "tab_ablation_merge_method.tex"
    standalone_path.write_text(_standalone(body, CAPTION, LABEL))
    print(f"  [saved] {standalone_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default=DEFAULT_INPUT,   type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    args = parser.parse_args()
    main(args.input, args.out_dir)
