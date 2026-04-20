"""Cumulative class counts per task for each dataset.

Used for x-axis labelling in line plots (number of classes seen so far).

Schedules derived from LAMDA-PILOT experiment configs:
  CIFAR100    : init=10,  inc=10,  T=10  → 10,20,...,100
  IN-R        : init=20,  inc=20,  T=10  → 20,40,...,200
  IN-A        : init=20,  inc=20,  T=10  → 20,40,...,200
  CUB         : init=20,  inc=20,  T=10  → 20,40,...,200
  OB          : init=30,  inc=30,  T=10  → 30,60,...,300
  VTAB        : init=10,  inc=10,  T=5   → 10,20,30,40,50
  CARS        : init=16,  inc=20,  T=10  → 16,36,56,...,196
"""

from __future__ import annotations


def _linear(init: int, inc: int, n_tasks: int) -> list[int]:
    return [init + inc * t for t in range(n_tasks)]


# Maps JSON dataset key -> cumulative class counts per task
SCHEDULES: dict[str, list[int]] = {
    "CIFAR":  _linear(10, 10, 10),
    "IN-R":   _linear(20, 20, 10),
    "IN-A":   _linear(20, 20, 10),
    "CUB":    _linear(20, 20, 10),
    "OB":     _linear(30, 30, 10),
    "VTAB":   _linear(10, 10, 5),
    "CARS":   _linear(16, 20, 10),
}

# Display names for axis labels
DATASET_DISPLAY: dict[str, str] = {
    "CIFAR":  "CIFAR100",
    "IN-R":   "IN-R",
    "IN-A":   "IN-A",
    "CUB":    "CUB",
    "OB":     "OB",
    "VTAB":   "VTAB",
    "CARS":   "CARS",
}


def get_schedule(dataset: str) -> list[int]:
    """Return cumulative class counts for a given dataset key."""
    if dataset not in SCHEDULES:
        raise KeyError(f"Unknown dataset '{dataset}'. Available: {list(SCHEDULES)}")
    return SCHEDULES[dataset]
