"""CIL evaluation metrics: FA, AA, FF.

All functions take `perf` — a lower-triangular list-of-lists where
  perf[t][i] = accuracy on task i evaluated after learning task t (0-indexed).

Metrics match LAMDA-PILOT / TUNA definition:
  A_t   = class-weighted total accuracy at stage t
         = sum_i(n_i * A_{t,i}) / sum_i(n_i)   [n_i = #classes in group i]
  FA    = A_T  (accuracy at final stage)
  AA    = (1/T) * sum_t A_t
  FF    = (1/(T-1)) * sum_{i<T} (max_t A_{t,i} - A_{T,i})

`group_sizes` must be provided for datasets with unequal group sizes (e.g.
CARS: [16, 20, 20, ...]). When None, falls back to simple per-group mean,
which is equivalent when all groups are equal-sized.
"""
from __future__ import annotations
import numpy as np


def _stage_acc(row: list[float], group_sizes: list[int] | None) -> float:
    """Class-weighted accuracy at one stage."""
    if group_sizes is None:
        return float(np.mean(row))
    w = group_sizes[:len(row)]
    return float(np.average(row, weights=w))


def compute_fa(perf: list[list[float]],
               group_sizes: list[int] | None = None) -> float:
    """Final Accuracy = class-weighted total accuracy after the last stage."""
    return _stage_acc(perf[-1], group_sizes)


def compute_aa(perf: list[list[float]],
               group_sizes: list[int] | None = None) -> float:
    """Accumulated Accuracy = mean of per-stage class-weighted accuracy."""
    return float(np.mean([_stage_acc(row, group_sizes) for row in perf]))


def compute_ff(perf: list[list[float]]) -> float:
    """Final Forgetting = mean over tasks i<T of (peak_i - final_i).
    Returns 0 when T < 2.
    """
    T = len(perf)
    if T < 2:
        return 0.0
    forgetting = []
    for i in range(T - 1):
        col = [perf[t][i] for t in range(i, T)]
        forgetting.append(max(col) - perf[-1][i])
    return float(np.mean(forgetting))


def aggregate_seeds(
    seed_perfs: list[list[list[float]]],
) -> dict[str, tuple[float, float]]:
    """Return {metric: (mean, std)} aggregated over a list of per-seed matrices."""
    fa = [compute_fa(p) for p in seed_perfs]
    aa = [compute_aa(p) for p in seed_perfs]
    ff = [compute_ff(p) for p in seed_perfs]
    return {
        "FA": (float(np.mean(fa)), float(np.std(fa))),
        "AA": (float(np.mean(aa)), float(np.std(aa))),
        "FF": (float(np.mean(ff)), float(np.std(ff))),
    }
