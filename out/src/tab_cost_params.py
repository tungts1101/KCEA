"""Generate trainable-parameter cost table.

Run from KCEA/:
    python -m out.src.tab_cost_params \\
        [--results result.json] [--costs costs.json] [--out-dir out/assets/tables]
"""
from __future__ import annotations
import argparse
from pathlib import Path

from out.src.cost_loader import load_costs, load_baseline_costs, merge_cost_sources
from out.src.cost_table import build_params_body, COST_BASELINE_KEYS, COST_KCEA_KEY

_HERE = Path(__file__).parent
DEFAULT_RESULTS = _HERE.parent.parent / "result.json"
DEFAULT_COSTS   = _HERE.parent.parent / "costs.json"
DEFAULT_OUT_DIR = _HERE.parent / "assets" / "tables"

CAPTION = (
    r"Trainable parameter cost comparison. "
    r"Each cell reports the mean $\pm$ std across datasets "
    r"(std computed over the per-dataset seed-means). "
    r"Best (lowest) per column in \textbf{bold}."
)
LABEL = "tab:cost-params"


def _standalone(body: str, caption: str, label: str) -> str:
    return "\n".join([
        r"\begin{table*}[!t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        body,
        r"\end{table*}",
    ])


def main(
    results_path: Path = DEFAULT_RESULTS,
    costs_path:   Path = DEFAULT_COSTS,
    out_dir:      Path = DEFAULT_OUT_DIR,
) -> str:
    baseline_costs = load_baseline_costs(results_path)
    kcea_costs     = load_costs(costs_path)
    costs          = merge_cost_sources(baseline_costs, kcea_costs)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    body = build_params_body(costs, COST_BASELINE_KEYS, COST_KCEA_KEY)

    body_path = out_dir / "tab_cost_params_body.tex"
    body_path.write_text(body)
    print(f"  [saved] {body_path}")

    standalone_path = out_dir / "tab_cost_params.tex"
    standalone_path.write_text(_standalone(body, CAPTION, LABEL))
    print(f"  [saved] {standalone_path}")
    return body


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=DEFAULT_RESULTS, type=Path)
    p.add_argument("--costs",   default=DEFAULT_COSTS,   type=Path)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    args = p.parse_args()
    main(args.results, args.costs, args.out_dir)
