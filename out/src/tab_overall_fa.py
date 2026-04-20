"""Write FA standalone table, body snippet, to assets/tables/.

Run from KCEA/:
    python -m out.src.tab_overall_fa [--input result.json] [--out-dir assets/tables]
"""
from __future__ import annotations
import argparse
from pathlib import Path

from out.src.loader import load_results, aggregate_all
from out.src.overall_table import build_body, ALL_KEYS, DATASETS_ORDER

_HERE = Path(__file__).parent
DEFAULT_INPUT   = _HERE.parent.parent / "result.json"
DEFAULT_OUT_DIR = _HERE.parent / "assets" / "tables"

CAPTION_STANDALONE = (
    r"Final Accuracy (FA, \%) on seven CIL benchmarks. "
    r"Each cell shows the mean across three seeds (1993, 1994, 1995) "
    r"with the standard deviation as a subscript. "
    r"Best result per column in \textbf{bold}, second best in \textit{italic}."
)
LABEL_STANDALONE = "tab:overall-fa-standalone"


def _standalone(body: str, caption: str, label: str) -> str:
    return "\n".join([
        r"\begin{table*}[!t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        body,
        r"\end{table*}",
    ])


def main(input_path: Path = DEFAULT_INPUT, out_dir: Path = DEFAULT_OUT_DIR) -> None:
    raw = load_results(input_path)
    agg = aggregate_all(raw, ALL_KEYS, DATASETS_ORDER)
    body = build_body(agg, metric="FA", lower_is_better=False)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # A — body-only snippet
    body_path = out_dir / "tab_overall_fa_body.tex"
    body_path.write_text(body)
    print(f"  [saved] {body_path}")

    # B — standalone table
    standalone_path = out_dir / "tab_overall_fa.tex"
    standalone_path.write_text(_standalone(body, CAPTION_STANDALONE, LABEL_STANDALONE))
    print(f"  [saved] {standalone_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default=DEFAULT_INPUT,   type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    args = parser.parse_args()
    main(args.input, args.out_dir)
