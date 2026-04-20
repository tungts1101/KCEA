"""Parse KCEA ablation log files into ablation.json.

Each log file contains multiple seeds.  The parser extracts the accuracy
matrix and configuration for every seed and writes structured output into
a JSON registry keyed by ablation_name → setting_value → seed.

Usage (run from KCEA/):
    python -m out.src.ablation_parser \\
        --logs logs_ablation/vtab/kcea_ablation__model_merge_incremental_True.log \\
               logs_ablation/vtab/kcea_ablation__model_merge_incremental_False.log \\
        --ablation-name merge_incremental \\
        --param model_merge_incremental \\
        --dataset vtab \\
        --output ablation.json

The existing ablation.json (if any) is updated in-place so multiple
ablations can be accumulated into one file.
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

# ── Log line prefix regex ─────────────────────────────────────────────────────
# Matches: "2026-04-18 15:24:53,823 [module.py] => content"
_LINE_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \S+ \[.*?\] => (.*)$")

# ── Matrix row regex ──────────────────────────────────────────────────────────
_ROW_RE = re.compile(r"^\s*\[[\d.,\s]+\]\s*$")


def _content(raw_line: str) -> str:
    """Strip timestamp + module prefix, return bare content (or '')."""
    m = _LINE_RE.match(raw_line.rstrip("\n"))
    return m.group(1) if m else ""


def _parse_matrix_row(content: str) -> list[float]:
    """'  [99.27, 64.28]' → [99.27, 64.28]"""
    return [float(x) for x in re.findall(r"[\d.]+", content)]


def parse_log(log_path: Path, param: str) -> list[dict]:
    """Parse a single KCEA log file.

    Returns a list of run dicts, each with keys:
        seed   : str
        setting: str   (value of `param` from the config block)
        config : dict  (all config key-value pairs)
        perf   : list[list[float]]  (accuracy matrix, lower-triangular)
    """
    runs: list[dict] = []
    current: dict | None = None
    in_config = False
    in_matrix = False
    last_experiment_line = ""

    with open(log_path, encoding="utf-8") as fh:
        for raw_line in fh:
            c = _content(raw_line)

            # ── "Starting experiment:" marks a run header ─────────────────────
            if "Starting experiment:" in c:
                last_experiment_line = c
                continue

            # ── "Configuration:" opens a new run ─────────────────────────────
            if c.strip() == "Configuration:":
                # Close previous run if complete
                if current is not None:
                    if current.get("perf"):
                        runs.append(current)
                    else:
                        warnings.warn(
                            f"[ablation_parser] Incomplete run (no perf matrix) "
                            f"at seed {current.get('seed')} — skipped."
                        )
                # Extract seed from last "Starting experiment:" line
                m = re.search(r"seed (\d+)", last_experiment_line)
                seed = m.group(1) if m else "unknown"
                current = {"seed": seed, "setting": None, "config": {}, "perf": []}
                in_config = True
                in_matrix = False
                continue

            # ── Inside config block ───────────────────────────────────────────
            if in_config:
                stripped = c.strip()
                if stripped.startswith("[") or stripped.startswith("=") or not c.startswith("  "):
                    # Config block ended (next training/storage line)
                    in_config = False
                elif ": " in stripped:
                    key, _, val = stripped.partition(": ")
                    current["config"][key.strip()] = val.strip()
                    if key.strip() == param:
                        current["setting"] = val.strip()
                continue

            # ── Accuracy matrix ───────────────────────────────────────────────
            if "[Evaluation] Accuracy matrix:" in c:
                in_matrix = True
                if current is not None:
                    current["perf"] = []
                continue

            if in_matrix and current is not None:
                stripped = c.strip()
                if stripped.startswith("[") and _ROW_RE.match(c):
                    current["perf"].append(_parse_matrix_row(c))
                else:
                    # Row ended (next [Evaluation] or separator line)
                    in_matrix = False

    # Close last run
    if current is not None:
        if current.get("perf"):
            runs.append(current)
        else:
            warnings.warn(
                f"[ablation_parser] Last run (seed {current.get('seed')}) "
                f"has no perf matrix — skipped."
            )

    return runs


def parse_logs(
    log_paths: list[Path],
    param: str,
) -> dict[str, dict[str, dict]]:
    """Parse a collection of log files.

    Returns:
        settings[setting_value][seed] = {"perf": ..., "config": ...}
    """
    settings: dict[str, dict[str, dict]] = {}
    for path in log_paths:
        runs = parse_log(path, param)
        if not runs:
            warnings.warn(f"[ablation_parser] No valid runs found in {path}.")
        for run in runs:
            sv = run["setting"]
            if sv is None:
                warnings.warn(
                    f"[ablation_parser] Run seed {run['seed']} in {path} "
                    f"has no value for param '{param}' — skipped."
                )
                continue
            settings.setdefault(sv, {})
            seed = run["seed"]
            if seed in settings[sv]:
                warnings.warn(
                    f"[ablation_parser] Duplicate seed {seed} for setting "
                    f"'{sv}' (from {path}) — overwriting."
                )
            settings[sv][seed] = {
                "performance": run["perf"],
                "config": run["config"],
            }
    return settings


def update_ablation_json(
    output_path: Path,
    ablation_name: str,
    param: str,
    dataset: str,
    settings: dict[str, dict[str, dict]],
) -> None:
    """Load (or create) ablation.json and update the entry for ablation_name."""
    if output_path.exists():
        with open(output_path) as fh:
            registry = json.load(fh)
    else:
        registry = {}

    registry[ablation_name] = {
        "param": param,
        "dataset": dataset,
        "settings": settings,
    }

    with open(output_path, "w") as fh:
        json.dump(registry, fh, indent=2)
    print(f"  [saved] {output_path}  ({len(settings)} settings)")


def main(
    logs: list[Path],
    ablation_name: str,
    param: str,
    dataset: str,
    output: Path,
) -> None:
    settings = parse_logs(logs, param)
    for sv, seeds in settings.items():
        print(
            f"  Setting '{param}={sv}': "
            f"{len(seeds)} seed(s) — {sorted(seeds.keys())}"
        )
    update_ablation_json(output, ablation_name, param, dataset, settings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse KCEA ablation logs → ablation.json")
    parser.add_argument(
        "--logs", nargs="+", type=Path, required=True,
        help="One or more KCEA log files to parse.",
    )
    parser.add_argument("--ablation-name", required=True, help="Registry key, e.g. merge_incremental")
    parser.add_argument("--param", required=True, help="Config key to distinguish settings, e.g. model_merge_incremental")
    parser.add_argument("--dataset", required=True, help="Dataset label stored in registry, e.g. vtab")
    parser.add_argument(
        "--output", type=Path, default=Path("ablation.json"),
        help="Output JSON path (default: ablation.json in cwd).",
    )
    args = parser.parse_args()
    main(args.logs, args.ablation_name, args.param, args.dataset, args.output)
