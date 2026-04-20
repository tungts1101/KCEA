"""Robust loader for result.json.

Loads the JSON and aggregates FA/AA/FF per (method, dataset) over available seeds.
Missing methods, datasets, or seeds produce warnings rather than crashes.

Public API
----------
load_results(path)  -> raw dict
aggregate_all(raw, methods, datasets) -> agg dict
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np

from out.src.metrics import compute_fa, compute_aa, compute_ff
from out.src.class_schedule import SCHEDULES


def _group_sizes(dataset: str) -> list[int] | None:
    """Per-group class counts for dataset, or None when all equal."""
    cum = SCHEDULES.get(dataset)
    if cum is None:
        return None
    sizes = [cum[0]] + [cum[t] - cum[t - 1] for t in range(1, len(cum))]
    return None if len(set(sizes)) == 1 else sizes

EXPECTED_SEEDS = 3


def load_results(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"result.json not found: {path}")
    with open(path) as f:
        return json.load(f)


def aggregate_all(
    raw: dict,
    methods: list[str],
    datasets: list[str],
) -> dict[str, dict[str, dict | None]]:
    """Aggregate metrics over seeds for every (method, dataset) pair.

    Returns
    -------
    agg[method][dataset] = {
        "FA": (mean, std),
        "AA": (mean, std),
        "FF": (mean, std),
        "n_seeds": int,
    }
    or None when no valid data exists.
    """
    agg: dict[str, dict[str, dict | None]] = {}
    for method in methods:
        agg[method] = {}
        method_data = raw.get(method, {})
        if not method_data:
            warnings.warn(f"[loader] Method '{method}' not found in results.")
        for dataset in datasets:
            seed_dict = method_data.get(dataset, {})
            if not seed_dict:
                warnings.warn(f"[loader] No data for {method}/{dataset}.")
                agg[method][dataset] = None
                continue

            gs = _group_sizes(dataset)
            fa_vals, aa_vals, ff_vals = [], [], []
            for seed, entry in seed_dict.items():
                perf = entry.get("performance", [])
                if not perf:
                    warnings.warn(
                        f"[loader] Empty performance matrix for "
                        f"{method}/{dataset}/seed {seed} — skipping."
                    )
                    continue
                fa_vals.append(compute_fa(perf, gs))
                aa_vals.append(compute_aa(perf, gs))
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
