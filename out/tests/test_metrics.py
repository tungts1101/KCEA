"""Unit tests for out/src/metrics.py.

Hand-crafted 3-task performance matrix:
  After task 0: [80.0]
  After task 1: [75.0, 90.0]
  After task 2: [70.0, 85.0, 95.0]

Expected values:
  FA  = (70 + 85 + 95) / 3              = 83.333...
  AA  = mean([80, (75+90)/2, (70+85+95)/3])
       = mean([80, 82.5, 83.333...])    = 81.944...
  FF  = mean([(80-70), (90-85)])        = 7.5
"""
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from out.src.metrics import compute_fa, compute_aa, compute_ff, aggregate_seeds

PERF = [
    [80.0],
    [75.0, 90.0],
    [70.0, 85.0, 95.0],
]


def test_fa():
    got = compute_fa(PERF)
    exp = (70 + 85 + 95) / 3
    assert math.isclose(got, exp, abs_tol=1e-9), f"FA: {got} != {exp}"


def test_aa():
    got = compute_aa(PERF)
    exp = (80.0 + (75 + 90) / 2 + (70 + 85 + 95) / 3) / 3
    assert math.isclose(got, exp, abs_tol=1e-9), f"AA: {got} != {exp}"


def test_ff():
    got = compute_ff(PERF)
    # task 0: peak=max(80,75,70)=80, final=70 → forgetting=10
    # task 1: peak=max(90,85)=90, final=85    → forgetting=5
    exp = (10 + 5) / 2
    assert math.isclose(got, exp, abs_tol=1e-9), f"FF: {got} != {exp}"


def test_ff_single_task():
    assert compute_ff([[90.0]]) == 0.0


def test_ff_no_regression():
    # Monotonically stable: peak == value at each step for task 0
    perf = [[90.0], [90.0, 92.0], [90.0, 92.0, 95.0]]
    assert math.isclose(compute_ff(perf), 0.0, abs_tol=1e-9)


def test_aggregate_seeds_zero_std():
    # Two identical seeds → std must be 0
    agg = aggregate_seeds([PERF, PERF])
    for key in ["FA", "AA", "FF"]:
        mean, std = agg[key]
        assert std == 0.0, f"{key} std={std} for identical seeds"


def test_aggregate_seeds_values():
    agg = aggregate_seeds([PERF])
    assert math.isclose(agg["FA"][0], (70 + 85 + 95) / 3, abs_tol=1e-9)
    assert math.isclose(agg["FF"][0], 7.5, abs_tol=1e-9)


if __name__ == "__main__":
    tests = [
        test_fa, test_aa, test_ff, test_ff_single_task,
        test_ff_no_regression, test_aggregate_seeds_zero_std,
        test_aggregate_seeds_values,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print("All tests passed.")
