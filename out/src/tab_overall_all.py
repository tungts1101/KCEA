"""Write the combined one-page wrapper and layout-verification document.

Outputs (all in assets/tables/):
  tab_overall_all.tex          — combined table* float with three subtables
  tab_overall_all_test.tex     — compilable standalone for layout verification

Also calls the three per-metric scripts to ensure body snippets are up to date.

Run from KCEA/:
    python -m out.src.tab_overall_all [--input result.json] [--out-dir assets/tables]
"""
from __future__ import annotations
import argparse
from pathlib import Path

from out.src.loader import load_results, aggregate_all
from out.src.overall_table import build_body, ALL_KEYS, DATASETS_ORDER

_HERE = Path(__file__).parent
DEFAULT_INPUT   = _HERE.parent.parent / "result.json"
DEFAULT_OUT_DIR = _HERE.parent / "assets" / "tables"

_COMBINED_CAPTION = (
    r"Main results on seven class-incremental benchmarks with ViT-B/16-IN1K. "
    r"Each cell reports the mean across three seeds (1993, 1994, 1995) "
    r"with standard deviation as a subscript. "
    r"Best result per column in \textbf{bold}, second best in \textit{italic}. "
    r"Lower is better for FF."
)

_COMBINED = r"""\begin{{table*}}[!t]
\centering
\caption{{{caption}}}
\label{{tab:overall}}

\begin{{subtable}}{{\textwidth}}
  \centering
  \caption{{Final Accuracy (FA, \%) -- higher is better.}}
  \label{{tab:overall-fa}}
  {body_fa}
\end{{subtable}}

\vspace{{0.6em}}

\begin{{subtable}}{{\textwidth}}
  \centering
  \caption{{Accumulated Accuracy (AA, \%) -- higher is better.}}
  \label{{tab:overall-aa}}
  {body_aa}
\end{{subtable}}

\vspace{{0.6em}}

\begin{{subtable}}{{\textwidth}}
  \centering
  \caption{{Final Forgetting (FF, \%) -- lower is better.}}
  \label{{tab:overall-ff}}
  {body_ff}
\end{{subtable}}
\end{{table*}}"""

_TEST_DOC = r"""\documentclass[10pt]{{article}}
\usepackage[a4paper, margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{adjustbox}}
\usepackage{{subcaption}}
\usepackage{{amsmath}}
\begin{{document}}
\input{{{combined_path}}}
\end{{document}}"""


def main(input_path: Path = DEFAULT_INPUT, out_dir: Path = DEFAULT_OUT_DIR) -> None:
    raw = load_results(input_path)
    agg = aggregate_all(raw, ALL_KEYS, DATASETS_ORDER)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build bodies and write body-only snippets
    bodies = {}
    for metric, lower in [("FA", False), ("AA", False), ("FF", True)]:
        body = build_body(agg, metric=metric, lower_is_better=lower)
        bodies[metric] = body
        body_path = out_dir / f"tab_overall_{metric.lower()}_body.tex"
        body_path.write_text(body)
        print(f"  [saved] {body_path}")

    # Combined wrapper — bodies inlined directly, no \input references
    combined_tex = _COMBINED.format(
        caption=_COMBINED_CAPTION,
        body_fa=bodies["FA"],
        body_aa=bodies["AA"],
        body_ff=bodies["FF"],
    )
    combined_path = out_dir / "tab_overall_all.tex"
    combined_path.write_text(combined_tex)
    print(f"  [saved] {combined_path}")

    # Layout-verification standalone document
    test_tex = _TEST_DOC.format(combined_path=combined_path.resolve())
    test_path = out_dir / "tab_overall_all_test.tex"
    test_path.write_text(test_tex)
    print(f"  [saved] {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default=DEFAULT_INPUT,   type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    args = parser.parse_args()
    main(args.input, args.out_dir)
