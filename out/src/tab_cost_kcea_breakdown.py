"""Generate per-dataset KCEA wall-clock breakdown table.

Run from KCEA/:
    python -m out.src.tab_cost_kcea_breakdown \\
        [--costs costs.json] [--method-key KCEA-FT] [--out-dir out/assets/tables]
"""
from __future__ import annotations
import argparse
from pathlib import Path

from out.src.cost_loader import load_costs
from out.src.cost_table import build_kcea_breakdown_body, COST_KCEA_KEY, DATASETS_ORDER

_HERE = Path(__file__).parent
DEFAULT_COSTS   = _HERE.parent.parent / "costs.json"
DEFAULT_OUT_DIR = _HERE.parent / "assets" / "tables"

# Fallback order: use most complete variant available
_KCEA_FALLBACK = ["KCEA-FT+Merge+Align", "KCEA-FT+Merge", "KCEA-FT", "KCEA"]

CAPTION = (
    r"Per-dataset wall-clock breakdown for KCEA. "
    r"Alignment time is the dedicated NES alignment overhead. "
    r"Times are mean $\pm$ std across three seeds."
)
LABEL = "tab:cost-kcea-breakdown"


def _standalone(body: str, caption: str, label: str) -> str:
    return "\n".join([
        r"\begin{table*}[!t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        body,
        r"\end{table*}",
    ])


def _best_available_key(costs: dict, preferred: str | None) -> str | None:
    if preferred and preferred in costs:
        return preferred
    for key in _KCEA_FALLBACK:
        if key in costs:
            return key
    return None


def main(
    costs_path: Path = DEFAULT_COSTS,
    method_key: str | None = None,
    out_dir:    Path = DEFAULT_OUT_DIR,
) -> str:
    costs = load_costs(costs_path)
    key   = _best_available_key(costs, method_key)

    if key is None:
        print("  [WARNING] No KCEA variant found in costs.json — skipping breakdown table.")
        return ""

    print(f"  Using method key: '{key}' for breakdown table.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    body = build_kcea_breakdown_body(costs, key, DATASETS_ORDER)

    body_path = out_dir / "tab_cost_kcea_breakdown_body.tex"
    body_path.write_text(body)
    print(f"  [saved] {body_path}")

    standalone_path = out_dir / "tab_cost_kcea_breakdown.tex"
    standalone_path.write_text(_standalone(body, CAPTION, LABEL))
    print(f"  [saved] {standalone_path}")
    return body


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--costs",      default=DEFAULT_COSTS,   type=Path)
    p.add_argument("--method-key", default=None,            type=str,
                   help="KCEA variant key in costs.json (default: best available).")
    p.add_argument("--out-dir",    default=DEFAULT_OUT_DIR, type=Path)
    args = p.parse_args()
    main(args.costs, args.method_key, args.out_dir)
