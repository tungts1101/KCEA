"""Unit tests for cost_loader functions."""
from __future__ import annotations
import json
import textwrap
from pathlib import Path

import pytest

from out.src.cost_loader import (
    load_costs,
    load_baseline_costs,
    merge_cost_sources,
    aggregate_across_datasets,
    aggregate_per_dataset,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

_COSTS_DATA = {
    "KCEA-FT": {
        "CIFAR": {
            "1993": {"training_s": 100.0, "nes_s": 5.0, "ram_owned_mb": 9.0},
            "1994": {"training_s": 110.0, "nes_s": 6.0, "ram_owned_mb": 9.2},
            "1995": {"training_s": 120.0, "nes_s": 7.0, "ram_owned_mb": 9.4},
        },
        "IN-R": {
            "1993": {"training_s": 200.0, "nes_s": 10.0, "ram_owned_mb": 9.1},
            "1994": {"training_s": 210.0, "nes_s": 11.0, "ram_owned_mb": 9.3},
        },
    }
}

_RESULT_DATA = {
    "SomeBaseline": {
        "CIFAR": {
            "1993": {
                "performance": [[0.9]],
                "config": {},
                "storage_cost": 15.0,
                "trainable_parameters": 1_000_000,
            },
            "1994": {
                "performance": [[0.85]],
                "config": {},
                "storage_cost": 16.0,
                "trainable_parameters": 1_000_000,
            },
        }
    },
    "NoFields": {
        "CIFAR": {
            "1993": {"performance": [[0.7]], "config": {}},
        }
    },
}


@pytest.fixture()
def costs_path(tmp_path: Path) -> Path:
    p = tmp_path / "costs.json"
    p.write_text(json.dumps(_COSTS_DATA))
    return p


@pytest.fixture()
def results_path(tmp_path: Path) -> Path:
    p = tmp_path / "result.json"
    p.write_text(json.dumps(_RESULT_DATA))
    return p


# ── load_costs ────────────────────────────────────────────────────────────────

class TestLoadCosts:
    def test_returns_dict(self, costs_path):
        data = load_costs(costs_path)
        assert isinstance(data, dict)

    def test_method_present(self, costs_path):
        data = load_costs(costs_path)
        assert "KCEA-FT" in data

    def test_dataset_present(self, costs_path):
        data = load_costs(costs_path)
        assert "CIFAR" in data["KCEA-FT"]

    def test_seed_present(self, costs_path):
        data = load_costs(costs_path)
        assert "1993" in data["KCEA-FT"]["CIFAR"]

    def test_field_value(self, costs_path):
        data = load_costs(costs_path)
        assert data["KCEA-FT"]["CIFAR"]["1993"]["training_s"] == pytest.approx(100.0)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_costs(tmp_path / "nonexistent.json")


# ── load_baseline_costs ───────────────────────────────────────────────────────

class TestLoadBaselineCosts:
    def test_returns_dict(self, results_path):
        data = load_baseline_costs(results_path)
        assert isinstance(data, dict)

    def test_method_with_cost_fields_included(self, results_path):
        data = load_baseline_costs(results_path)
        assert "SomeBaseline" in data

    def test_method_without_cost_fields_excluded(self, results_path):
        data = load_baseline_costs(results_path)
        assert "NoFields" not in data

    def test_cost_field_value(self, results_path):
        data = load_baseline_costs(results_path)
        assert data["SomeBaseline"]["CIFAR"]["1993"]["storage_cost"] == pytest.approx(15.0)

    def test_missing_results_file_returns_empty(self, tmp_path):
        data = load_baseline_costs(tmp_path / "nonexistent.json")
        assert data == {}


# ── merge_cost_sources ────────────────────────────────────────────────────────

class TestMergeCostSources:
    def test_merges_disjoint_methods(self):
        a = {"M1": {"DS1": {"s1": {"f": 1.0}}}}
        b = {"M2": {"DS2": {"s2": {"f": 2.0}}}}
        merged = merge_cost_sources(a, b)
        assert "M1" in merged and "M2" in merged

    def test_kcea_takes_precedence(self):
        baseline = {"KCEA-FT": {"CIFAR": {"1993": {"training_s": 999.0}}}}
        kcea     = {"KCEA-FT": {"CIFAR": {"1993": {"training_s": 100.0}}}}
        merged = merge_cost_sources(baseline, kcea)
        assert merged["KCEA-FT"]["CIFAR"]["1993"]["training_s"] == pytest.approx(100.0)

    def test_baseline_seeds_preserved(self):
        baseline = {"M": {"DS": {"s1": {"f": 1.0}, "s2": {"f": 2.0}}}}
        kcea     = {"M": {"DS": {"s3": {"f": 3.0}}}}
        merged = merge_cost_sources(baseline, kcea)
        assert set(merged["M"]["DS"].keys()) == {"s1", "s2", "s3"}

    def test_empty_inputs(self):
        assert merge_cost_sources({}, {}) == {}
        assert merge_cost_sources({"M": {}}, {}) == {"M": {}}


# ── aggregate_across_datasets ─────────────────────────────────────────────────

class TestAggregateAcrossDatasets:
    def test_mean_of_dataset_means(self, costs_path):
        costs = load_costs(costs_path)
        # CIFAR: mean(100, 110, 120) = 110.0
        # IN-R:  mean(200, 210)      = 205.0
        # grand mean: (110 + 205) / 2 = 157.5
        mean, std = aggregate_across_datasets(costs, "KCEA-FT", ["CIFAR", "IN-R"], "training_s")
        assert mean == pytest.approx(157.5, abs=1e-9)

    def test_std_of_dataset_means(self, costs_path):
        costs = load_costs(costs_path)
        # std([110, 205]) = std of two values
        import numpy as np
        expected_std = float(np.std([110.0, 205.0]))
        _, std = aggregate_across_datasets(costs, "KCEA-FT", ["CIFAR", "IN-R"], "training_s")
        assert std == pytest.approx(expected_std, abs=1e-9)

    def test_missing_method_returns_none(self, costs_path):
        costs = load_costs(costs_path)
        mean, std = aggregate_across_datasets(costs, "NONEXISTENT", ["CIFAR"], "training_s")
        assert mean is None and std is None

    def test_missing_field_returns_none(self, costs_path):
        costs = load_costs(costs_path)
        mean, std = aggregate_across_datasets(costs, "KCEA-FT", ["CIFAR"], "nonexistent_field")
        assert mean is None and std is None

    def test_partial_datasets(self, costs_path):
        costs = load_costs(costs_path)
        # Only CIFAR has data; VTAB is absent → only CIFAR contributes
        mean, std = aggregate_across_datasets(costs, "KCEA-FT", ["CIFAR", "VTAB"], "training_s")
        assert mean == pytest.approx(110.0, abs=1e-9)
        assert std == pytest.approx(0.0, abs=1e-9)


# ── aggregate_per_dataset ─────────────────────────────────────────────────────

class TestAggregatePerDataset:
    def test_returns_dict_of_tuples(self, costs_path):
        costs = load_costs(costs_path)
        result = aggregate_per_dataset(costs, "KCEA-FT", "CIFAR", ["training_s", "nes_s"])
        assert isinstance(result, dict)
        assert "training_s" in result and "nes_s" in result

    def test_mean_value(self, costs_path):
        costs = load_costs(costs_path)
        result = aggregate_per_dataset(costs, "KCEA-FT", "CIFAR", ["training_s"])
        mean, _ = result["training_s"]
        assert mean == pytest.approx(110.0, abs=1e-9)  # (100+110+120)/3

    def test_std_value(self, costs_path):
        import numpy as np
        costs = load_costs(costs_path)
        result = aggregate_per_dataset(costs, "KCEA-FT", "CIFAR", ["training_s"])
        _, std = result["training_s"]
        expected = float(np.std([100.0, 110.0, 120.0]))
        assert std == pytest.approx(expected, abs=1e-9)

    def test_missing_field_returns_none_tuple(self, costs_path):
        costs = load_costs(costs_path)
        result = aggregate_per_dataset(costs, "KCEA-FT", "CIFAR", ["absent_field"])
        assert result["absent_field"] == (None, None)

    def test_two_seeds(self, costs_path):
        import numpy as np
        costs = load_costs(costs_path)
        result = aggregate_per_dataset(costs, "KCEA-FT", "IN-R", ["nes_s"])
        mean, std = result["nes_s"]
        assert mean == pytest.approx(float(np.mean([10.0, 11.0])), abs=1e-9)
