"""Parse KCEA log resource reports into costs.json.

Each log file contains END-OF-RUN RESOURCE REPORT blocks (one per seed).
This parser extracts all numeric fields and writes a structured JSON registry.

Usage (run from KCEA/):
    python -m out.src.cost_parser \\
        --logs logs/cifar224/kcea_ft.log logs/imagenetr/kcea_ft.log ... \\
        --output costs.json

costs.json schema:
    {
      "KCEA-FT": {
        "CIFAR": {
          "1993": {resource fields...},
          "1994": {...},
          "1995": {...}
        },
        ...
      }
    }

Idempotent: re-running merges new data into existing entries.
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

# ── Log line prefix stripping ─────────────────────────────────────────────────
_LINE_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \S+ \[.*?\] => (.*)$")


def _content(raw_line: str) -> str:
    m = _LINE_RE.match(raw_line.rstrip("\n"))
    return m.group(1) if m else raw_line.rstrip("\n")


# ── Name maps ─────────────────────────────────────────────────────────────────
DATASET_NAME_MAP: dict[str, str] = {
    "cifar224":      "CIFAR",
    "imagenetr":     "IN-R",
    "imageneta":     "IN-A",
    "cub":           "CUB",
    "omnibenchmark": "OB",
    "vtab":          "VTAB",
    "cars":          "CARS",
}

# train_prefix (or Method field) → canonical method key
# Longer prefixes must come first to avoid partial matches
PREFIX_TO_METHOD: dict[str, str] = {
    "kcea_ft_merge_align": "KCEA-FT+Merge+Align",
    "kcea_ft_merge":       "KCEA-FT+Merge",
    "kcea_ft":             "KCEA-FT",
    "kcea_full":           "KCEA",
    "kcea":                "KCEA",
}

# ── Resource field patterns ───────────────────────────────────────────────────
# Each entry: (field_name, compiled_regex, cast_fn)
_FIELD_PATTERNS: list[tuple[str, re.Pattern, object]] = [
    ("wall_clock_s",
     re.compile(r"Wall-clock time:\s*([\d.]+)s"), float),
    ("training_s",
     re.compile(r"Training \(SGD.*?\)\s*:\s*([\d.]+)s"), float),
    ("merging_s",
     re.compile(r"Parameter merging.*?:\s*([\d.]+)s"), float),
    ("gaussian_s",
     re.compile(r"Gaussian statistics \(per task\)\s*:\s*([\d.]+)s"), float),
    ("nes_s",
     re.compile(r"NES alignment.*?:\s*([\d.]+)s"), float),
    ("evaluation_s",
     re.compile(r"Evaluation / test.*?:\s*([\d.]+)s"), float),
    ("data_loading_s",
     re.compile(r"Data loading.*?:\s*([\d.]+)s"), float),
    ("active_compute_s",
     re.compile(r"Active compute.*?:\s*([\d.]+)s"), float),
    ("ram_owned_mb",
     re.compile(r"Actual RAM owned by KCEA\s*:\s*([\d.]+)\s*MB"), float),
    ("ram_peft_mb",
     re.compile(r"PEFT adapter params\s*:\s*([\d.]+)\s*MB"), float),
    ("ram_classifier_mb",
     re.compile(r"Classifier heads.*?:\s*([\d.]+)\s*MB"), float),
    ("ram_gaussian_mb",
     re.compile(r"Gaussian statistics \(means.*?\)\s*:\s*([\d.]+)\s*MB"), float),
    ("ram_backbone_mb",
     re.compile(r"Frozen pretrained backbone\s*:\s*([\d.]+)\s*MB"), float),
    ("disk_total_mb",
     re.compile(r"Total written to disk.*?:\s*([\d.]+)\s*MB"), float),
    ("trainable_params_peak",
     re.compile(r"Peak trainable.*?:\s*([\d,]+)"),
     lambda x: int(x.replace(",", ""))),
    ("trainable_params_final",
     re.compile(r"Final task trainable\s*:\s*([\d,]+)"),
     lambda x: int(x.replace(",", ""))),
    ("total_params_final",
     re.compile(r"Total model params.*?:\s*([\d,]+)"),
     lambda x: int(x.replace(",", ""))),
]

# Line that ends a resource report block
_SEPARATOR_RE = re.compile(r"^={10,}$")


def _parse_resource_fields(lines: list[str]) -> dict:
    """Extract all resource fields from a collected list of [Summary] lines."""
    text = "\n".join(lines)
    result: dict = {}
    for field, pattern, cast in _FIELD_PATTERNS:
        m = pattern.search(text)
        if m:
            try:
                result[field] = cast(m.group(1))
            except (ValueError, TypeError):
                pass
    return result


def _method_key(raw: str) -> str | None:
    """Map a raw method string (train_prefix or report Method field) to canonical key."""
    for prefix, key in PREFIX_TO_METHOD.items():
        if raw == prefix or raw.startswith(prefix + "_"):
            return key
    # Exact match fallback (e.g., 'kcea' already handled above)
    return None


def parse_resource_report(log_path: str | Path) -> dict:
    """Parse all resource report blocks from a single KCEA log file.

    Returns
    -------
    {method_key: {dataset_key: {seed_str: resource_dict}}}
    """
    log_path = Path(log_path)
    result: dict = {}

    in_report = False
    meta: dict = {}
    report_lines: list[str] = []

    with open(log_path, encoding="utf-8") as fh:
        for raw_line in fh:
            c = _content(raw_line)

            # ── Report start ─────────────────────────────────────────────────
            if "╔══ END-OF-RUN RESOURCE REPORT ══╗" in c:
                in_report = True
                meta = {}
                report_lines = []
                continue

            if in_report:
                # ── Separator = report end ────────────────────────────────
                if _SEPARATOR_RE.match(c.strip()):
                    if meta and report_lines:
                        fields = _parse_resource_fields(report_lines)
                        _store(result, meta, fields)
                    in_report = False
                    meta = {}
                    report_lines = []
                    continue

                # ── Meta line: Method | Dataset | Seed ───────────────────
                if "Method :" in c and "Dataset :" in c and "Seed :" in c:
                    m = re.search(
                        r"Method\s*:\s*(\S+).*?Dataset\s*:\s*(\S+).*?Seed\s*:\s*(\d+)",
                        c,
                    )
                    if m:
                        meta = {
                            "method_raw":  m.group(1).strip(),
                            "dataset_raw": m.group(2).strip(),
                            "seed":        m.group(3).strip(),
                        }

                report_lines.append(c)

    # Handle file that ends without final separator
    if in_report and meta and report_lines:
        fields = _parse_resource_fields(report_lines)
        _store(result, meta, fields)

    return result


def _store(result: dict, meta: dict, fields: dict) -> None:
    method_raw  = meta.get("method_raw", "")
    dataset_raw = meta.get("dataset_raw", "")
    seed        = meta.get("seed", "unknown")

    method_key = _method_key(method_raw)
    if method_key is None:
        warnings.warn(
            f"[cost_parser] Unknown method '{method_raw}' — skipped. "
            f"Add it to PREFIX_TO_METHOD if needed."
        )
        return

    dataset_key = DATASET_NAME_MAP.get(dataset_raw)
    if dataset_key is None:
        warnings.warn(f"[cost_parser] Unknown dataset '{dataset_raw}' — skipped.")
        return

    if not fields:
        warnings.warn(
            f"[cost_parser] No resource fields extracted for "
            f"{method_key}/{dataset_key}/seed {seed}."
        )
        return

    result.setdefault(method_key, {}).setdefault(dataset_key, {})[seed] = fields


def update_costs_json(output_path: Path, parsed: dict) -> None:
    """Merge parsed data into existing costs.json (idempotent)."""
    if output_path.exists():
        with open(output_path) as fh:
            registry = json.load(fh)
    else:
        registry = {}

    for method, ds_dict in parsed.items():
        registry.setdefault(method, {})
        for ds, seed_dict in ds_dict.items():
            registry[method].setdefault(ds, {})
            registry[method][ds].update(seed_dict)

    with open(output_path, "w") as fh:
        json.dump(registry, fh, indent=2)


def main(logs: list[Path], output: Path) -> None:
    output = Path(output)
    all_parsed: dict = {}

    for log_path in logs:
        log_path = Path(log_path)
        if not log_path.exists():
            warnings.warn(f"[cost_parser] Log not found: {log_path}")
            continue

        parsed = parse_resource_report(log_path)
        for method, ds_dict in parsed.items():
            all_parsed.setdefault(method, {})
            for ds, seed_dict in ds_dict.items():
                all_parsed[method].setdefault(ds, {})
                for seed, fields in seed_dict.items():
                    all_parsed[method][ds][seed] = fields
                    print(f"  {method}/{ds}/seed {seed}: "
                          f"training={fields.get('training_s')}s, "
                          f"nes={fields.get('nes_s')}s, "
                          f"ram_owned={fields.get('ram_owned_mb')}MB")

    update_costs_json(output, all_parsed)
    print(f"  [saved] {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse KCEA log resource reports → costs.json"
    )
    parser.add_argument("--logs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("costs.json"),
        help="Output path (default: costs.json in cwd)",
    )
    args = parser.parse_args()
    main(args.logs, args.output)
