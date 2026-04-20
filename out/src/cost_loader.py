"""Load and merge cost data from costs.json and result.json.

Public API
----------
load_costs(path)                                -> raw costs.json dict
load_baseline_costs(results_path)              -> cost fields from result.json (if present)
merge_cost_sources(baseline_costs, kcea_costs) -> unified cost dict
aggregate_across_datasets(costs, method, datasets, field)
                                               -> (mean_of_ds_means, std_of_ds_means)
aggregate_per_dataset(costs, method, dataset, fields)
                                               -> {field: (mean, std)} over seeds
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

# Fields that *could* exist in result.json per-seed entries (currently absent;
# kept here so load_baseline_costs is forward-compatible).
_RESULT_COST_FIELDS = (
    "storage_cost",
    "trainable_parameters",
    "total_parameters",
    "running_time",
)


def load_costs(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"costs.json not found: {path}")
    with open(path) as fh:
        return json.load(fh)


def load_baseline_costs(results_path: str | Path) -> dict:
    """Extract cost fields from result.json for baseline methods.

    result.json currently stores only 'performance' and 'config' per seed,
    so this returns an empty dict.  Once cost fields are appended to
    result.json entries this function will populate them automatically.

    Returns
    -------
    Same schema as costs.json:
        {method: {dataset: {seed: resource_dict}}}
    """
    results_path = Path(results_path)
    if not results_path.exists():
        return {}

    with open(results_path) as fh:
        raw = json.load(fh)

    costs: dict = {}
    for method, ds_dict in raw.items():
        for ds, seed_dict in ds_dict.items():
            for seed, run in seed_dict.items():
                cost_fields = {
                    k: run[k] for k in _RESULT_COST_FIELDS if k in run
                }
                if cost_fields:
                    costs.setdefault(method, {}).setdefault(ds, {})[seed] = cost_fields

    return costs


def merge_cost_sources(baseline_costs: dict, kcea_costs: dict) -> dict:
    """Merge baseline (result.json) and KCEA (costs.json) cost dicts.

    kcea_costs takes precedence for any overlapping method/dataset/seed keys.
    """
    merged: dict = {}
    for source in (baseline_costs, kcea_costs):
        for method, ds_dict in source.items():
            merged.setdefault(method, {})
            for ds, seed_dict in ds_dict.items():
                merged[method].setdefault(ds, {})
                merged[method][ds].update(seed_dict)
    return merged


def _seed_vals(costs: dict, method: str, dataset: str, field: str) -> list[float]:
    """Return non-None float values for (method, dataset, field) across all seeds."""
    vals = []
    for resource in costs.get(method, {}).get(dataset, {}).values():
        v = resource.get(field)
        if v is not None:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
    return vals


def aggregate_across_datasets(
    costs: dict,
    method: str,
    datasets: list[str],
    field: str,
) -> tuple[float | None, float | None]:
    """Compute mean ± std where std is across per-dataset seed-means.

    Step 1: per dataset → mean over seeds.
    Step 2: mean ± std of those per-dataset means.
    """
    ds_means: list[float] = []
    for ds in datasets:
        vals = _seed_vals(costs, method, ds, field)
        if vals:
            ds_means.append(float(np.mean(vals)))
    if not ds_means:
        return None, None
    return float(np.mean(ds_means)), float(np.std(ds_means))


def aggregate_per_dataset(
    costs: dict,
    method: str,
    dataset: str,
    fields: list[str],
) -> dict[str, tuple[float | None, float | None]]:
    """Return {field: (mean, std)} aggregated over seeds for one (method, dataset)."""
    result: dict = {}
    for field in fields:
        vals = _seed_vals(costs, method, dataset, field)
        if vals:
            result[field] = (float(np.mean(vals)), float(np.std(vals)))
        else:
            result[field] = (None, None)
    return result
