"""Load ablation.json and aggregate FA/AA/FF per (setting, seed-group).

Public API
----------
load_ablation(path)           -> raw dict (full registry)
aggregate_ablation(entry)     -> dict[setting_value, {FA, AA, FF, trajectories}]
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from out.src.metrics import compute_fa, compute_aa, compute_ff


def load_ablation(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ablation.json not found: {path}")
    with open(path) as fh:
        return json.load(fh)


def aggregate_ablation(entry: dict) -> dict[str, dict]:
    """Aggregate metrics for one ablation entry.

    Parameters
    ----------
    entry : ablation registry entry with keys 'param', 'dataset', 'settings'

    Returns
    -------
    agg[setting_value] = {
        "FA"          : (mean, std),
        "AA"          : (mean, std),
        "FF"          : (mean, std),
        "n_seeds"     : int,
        "trajectories": list[list[float]]  # per-seed per-task accuracy (mean of row)
    }
    """
    agg: dict[str, dict] = {}
    settings = entry.get("settings", {})

    for sv, seed_dict in settings.items():
        fa_vals, aa_vals, ff_vals = [], [], []
        trajectories: list[list[float]] = []

        for seed, run in seed_dict.items():
            perf = run.get("performance", [])
            if not perf:
                warnings.warn(f"[ablation_loader] Empty perf for setting={sv} seed={seed}")
                continue
            fa_vals.append(compute_fa(perf))
            aa_vals.append(compute_aa(perf))
            ff_vals.append(compute_ff(perf))
            # Trajectory: per-task step accuracy = mean of row t (all seen tasks)
            trajectories.append([float(np.mean(row)) for row in perf])

        if not fa_vals:
            warnings.warn(f"[ablation_loader] No valid seeds for setting={sv}")
            agg[sv] = None
            continue

        agg[sv] = {
            "FA": (float(np.mean(fa_vals)), float(np.std(fa_vals))),
            "AA": (float(np.mean(aa_vals)), float(np.std(aa_vals))),
            "FF": (float(np.mean(ff_vals)), float(np.std(ff_vals))),
            "n_seeds": len(fa_vals),
            "trajectories": trajectories,
        }

    return agg
