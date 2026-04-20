"""Combined cost table wrapper (params + memory + time as subtables).

Outputs:
    tab_cost_all.tex          — combined table* float, bodies inlined
    tab_cost_all_test.tex     — compilable standalone for layout verification

Run from KCEA/:
    python -m out.src.tab_cost_all \\
        [--results result.json] [--costs costs.json] [--out-dir out/assets/tables]
"""
from __future__ import annotations
import argparse
from pathlib import Path

from out.src.cost_loader import load_costs, load_baseline_costs, merge_cost_sources
from out.src.cost_table import (
    build_params_body, build_memory_body, build_time_body,
    COST_BASELINE_KEYS, COST_KCEA_KEY,
)

_HERE = Path(__file__).parent
DEFAULT_RESULTS = _HERE.parent.parent / "result.json"
DEFAULT_COSTS   = _HERE.parent.parent / "costs.json"
DEFAULT_OUT_DIR = _HERE.parent / "assets" / "tables"

_COMBINED_CAPTION = (
    r"Cost analysis comparing KCEA against baselines on seven benchmarks. "
    r"Each cell reports the mean $\pm$ std across datasets. "
    r"Best (lowest) per column in \textbf{bold}. "
    r"Alignment time is shown only for KCEA; baselines have no alignment step."
)

_COMBINED = r"""\begin{{table*}}[!t]
\centering
\caption{{{caption}}}
\label{{tab:cost-all}}

\begin{{subtable}}{{\textwidth}}
  \centering
  \caption{{Trainable parameters.}}
  \label{{tab:cost-params}}
  {body_params}
\end{{subtable}}

\vspace{{0.6em}}

\begin{{subtable}}{{\textwidth}}
  \centering
  \caption{{Memory cost (MB).}}
  \label{{tab:cost-memory}}
  {body_memory}
\end{{subtable}}

\vspace{{0.6em}}

\begin{{subtable}}{{\textwidth}}
  \centering
  \caption{{Wall-clock time (s, cumulative across all tasks).}}
  \label{{tab:cost-time}}
  {body_time}
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


def main(
    results_path: Path = DEFAULT_RESULTS,
    costs_path:   Path = DEFAULT_COSTS,
    out_dir:      Path = DEFAULT_OUT_DIR,
) -> None:
    baseline_costs = load_baseline_costs(results_path)
    kcea_costs     = load_costs(costs_path)
    costs          = merge_cost_sources(baseline_costs, kcea_costs)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    body_params = build_params_body(costs, COST_BASELINE_KEYS, COST_KCEA_KEY)
    body_memory = build_memory_body(costs, COST_BASELINE_KEYS, COST_KCEA_KEY)
    body_time   = build_time_body(costs, COST_BASELINE_KEYS, COST_KCEA_KEY)

    # Write individual body-only snippets
    for name, body in [
        ("tab_cost_params_body.tex",  body_params),
        ("tab_cost_memory_body.tex",  body_memory),
        ("tab_cost_time_body.tex",    body_time),
    ]:
        path = out_dir / name
        path.write_text(body)
        print(f"  [saved] {path}")

    # Combined wrapper (bodies inlined)
    combined_tex = _COMBINED.format(
        caption=_COMBINED_CAPTION,
        body_params=body_params,
        body_memory=body_memory,
        body_time=body_time,
    )
    combined_path = out_dir / "tab_cost_all.tex"
    combined_path.write_text(combined_tex)
    print(f"  [saved] {combined_path}")

    # Layout-verification standalone
    test_tex = _TEST_DOC.format(combined_path=combined_path.resolve())
    test_path = out_dir / "tab_cost_all_test.tex"
    test_path.write_text(test_tex)
    print(f"  [saved] {test_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=DEFAULT_RESULTS, type=Path)
    p.add_argument("--costs",   default=DEFAULT_COSTS,   type=Path)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    args = p.parse_args()
    main(args.results, args.costs, args.out_dir)
