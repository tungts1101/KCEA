"""Unit tests for cost_parser.parse_resource_report."""
from __future__ import annotations
import textwrap
from pathlib import Path

import pytest

from out.src.cost_parser import parse_resource_report


_SYNTHETIC_LOG = textwrap.dedent("""\
    2026-01-01 00:00:00,000 [learner.py] => Starting experiment: vtab - kcea_ft - seed 1993
    2026-01-01 00:00:00,001 [learner.py] => Configuration:
    2026-01-01 00:00:00,001 [learner.py] =>   seed: 1993
    2026-01-01 00:00:00,001 [learner.py] =>   train_prefix: kcea_ft
    2026-01-01 00:00:00,001 [learner.py] =>   dataset_name: vtab
    2026-01-01 00:00:01,000 [learner.py] => [Training] Task 0
    2026-01-01 00:02:00,000 [learner.py] => [Summary] ╔══ END-OF-RUN RESOURCE REPORT ══╗
    2026-01-01 00:02:00,000 [learner.py] => [Summary]   Method : kcea_ft  |  Dataset : vtab  |  Seed : 1993
    2026-01-01 00:02:00,000 [learner.py] => [Summary]
    2026-01-01 00:02:00,000 [learner.py] => [Summary] ── Wall-clock time: 120.5s ──────────────────────────
    2026-01-01 00:02:00,000 [learner.py] => [Summary]   Training (SGD, all tasks)              : 75.3s  (62.5%)
    2026-01-01 00:02:00,000 [learner.py] => [Summary]   Parameter merging (TIES)               : 0.2s  (0.2%)
    2026-01-01 00:02:00,000 [learner.py] => [Summary]   Gaussian statistics (per task)         : 0.0s  (0.0%)
    2026-01-01 00:02:00,000 [learner.py] => [Summary]   NES alignment (per task)               : 5.1s  (4.2%)
    2026-01-01 00:02:00,000 [learner.py] => [Summary]   Evaluation / test inference            : 39.8s  (33.0%)
    2026-01-01 00:02:00,000 [learner.py] => [Summary]   Data loading (train batches)           : 22.4s  (18.6%)
    2026-01-01 00:02:00,000 [learner.py] => [Summary]   Other (I/O, misc)                      : 0.1s  (0.1%)
    2026-01-01 00:02:00,000 [learner.py] => [Summary]   ──────────────────────────────────────────────────────────────
    2026-01-01 00:02:00,000 [learner.py] => [Summary]   Active compute (wall − eval − data)    : 58.3s  (48.4%)
    2026-01-01 00:02:00,000 [learner.py] => [Summary]
    2026-01-01 00:02:00,000 [learner.py] => [Summary] ── Storage ─────────────────────────────────────────────────────
    2026-01-01 00:02:00,000 [learner.py] => [Summary]   Model weights in RAM (final task)      : 336.4 MB  [baseline-comparable]
    2026-01-01 00:02:00,000 [learner.py] => [Summary]     Frozen pretrained backbone              : 327.3 MB  (shared, not stored by KCEA)
    2026-01-01 00:02:00,000 [learner.py] => [Summary]     PEFT adapter params                     : 9.0 MB
    2026-01-01 00:02:00,000 [learner.py] => [Summary]     Classifier heads (all tasks)            : 0.1 MB
    2026-01-01 00:02:00,000 [learner.py] => [Summary]   Actual RAM owned by KCEA                : 9.2 MB  [PEFT + heads + statistics]
    2026-01-01 00:02:00,000 [learner.py] => [Summary]     Gaussian statistics (means + covs)      : 0.1 MB  (50 classes × 768-dim)
    2026-01-01 00:02:00,000 [learner.py] => [Summary]   Total written to disk (training)       : 54.1 MB
    2026-01-01 00:02:00,000 [learner.py] => [Summary]
    2026-01-01 00:02:00,000 [learner.py] => [Summary] ── Trainable parameters ─────────────────────────────────────────
    2026-01-01 00:02:00,000 [learner.py] => [Summary]   Total model params (final task)        : 88,197,888
    2026-01-01 00:02:00,000 [learner.py] => [Summary]   Peak trainable (any task)              : 2,366,976  (2.68% of 88,167,168 at that task)
    2026-01-01 00:02:00,000 [learner.py] => [Summary]   Final task trainable                   : 2,366,976  (2.68% of total at final task)
    2026-01-01 00:02:00,000 [learner.py] => [Summary]
    2026-01-01 00:02:00,001 [learner.py] => ================================================================================
    2026-01-01 00:02:00,002 [learner.py] => Experiment vtab_kcea_ft_seed1993 time: 120.50s
""")


@pytest.fixture()
def log_path(tmp_path: Path) -> Path:
    p = tmp_path / "kcea_ft.log"
    p.write_text(_SYNTHETIC_LOG)
    return p


class TestParseResourceReport:
    def test_one_run_extracted(self, log_path):
        result = parse_resource_report(log_path)
        assert "KCEA-FT" in result
        assert "VTAB" in result["KCEA-FT"]
        assert "1993" in result["KCEA-FT"]["VTAB"]

    def test_wall_clock(self, log_path):
        r = parse_resource_report(log_path)["KCEA-FT"]["VTAB"]["1993"]
        assert r["wall_clock_s"] == pytest.approx(120.5, abs=1e-9)

    def test_training_time(self, log_path):
        r = parse_resource_report(log_path)["KCEA-FT"]["VTAB"]["1993"]
        assert r["training_s"] == pytest.approx(75.3, abs=1e-9)

    def test_nes_time(self, log_path):
        r = parse_resource_report(log_path)["KCEA-FT"]["VTAB"]["1993"]
        assert r["nes_s"] == pytest.approx(5.1, abs=1e-9)

    def test_evaluation_time(self, log_path):
        r = parse_resource_report(log_path)["KCEA-FT"]["VTAB"]["1993"]
        assert r["evaluation_s"] == pytest.approx(39.8, abs=1e-9)

    def test_ram_peft(self, log_path):
        r = parse_resource_report(log_path)["KCEA-FT"]["VTAB"]["1993"]
        assert r["ram_peft_mb"] == pytest.approx(9.0, abs=1e-9)

    def test_ram_owned(self, log_path):
        r = parse_resource_report(log_path)["KCEA-FT"]["VTAB"]["1993"]
        assert r["ram_owned_mb"] == pytest.approx(9.2, abs=1e-9)

    def test_ram_gaussian(self, log_path):
        r = parse_resource_report(log_path)["KCEA-FT"]["VTAB"]["1993"]
        assert r["ram_gaussian_mb"] == pytest.approx(0.1, abs=1e-9)

    def test_disk_total(self, log_path):
        r = parse_resource_report(log_path)["KCEA-FT"]["VTAB"]["1993"]
        assert r["disk_total_mb"] == pytest.approx(54.1, abs=1e-9)

    def test_total_params(self, log_path):
        r = parse_resource_report(log_path)["KCEA-FT"]["VTAB"]["1993"]
        assert r["total_params_final"] == 88_197_888

    def test_peak_trainable(self, log_path):
        r = parse_resource_report(log_path)["KCEA-FT"]["VTAB"]["1993"]
        assert r["trainable_params_peak"] == 2_366_976

    def test_final_trainable(self, log_path):
        r = parse_resource_report(log_path)["KCEA-FT"]["VTAB"]["1993"]
        assert r["trainable_params_final"] == 2_366_976

    def test_merging_time(self, log_path):
        r = parse_resource_report(log_path)["KCEA-FT"]["VTAB"]["1993"]
        assert r["merging_s"] == pytest.approx(0.2, abs=1e-9)
