"""Unit tests for ablation_parser.parse_log / parse_logs."""
from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path

import pytest

from out.src.ablation_parser import parse_log, parse_logs


# ── Minimal synthetic log content ─────────────────────────────────────────────
# Two seeds in a single file, setting value 'True'.

_LOG_INCREMENTAL = textwrap.dedent("""\
    2026-01-01 00:00:00,000 [learner.py] => \

    ================================================================================
    2026-01-01 00:00:00,000 [learner.py] => Starting experiment: vtab - kcea_ablation__model_merge_incremental_True - seed 1993
    2026-01-01 00:00:01,000 [learner.py] => \

    ================================================================================
    2026-01-01 00:00:01,000 [learner.py] => Starting experiment: vtab - kcea_ablation__model_merge_incremental_True - seed 1993
    2026-01-01 00:00:01,001 [learner.py] => Dataset Training Size: 1000
    2026-01-01 00:00:01,001 [learner.py] => Configuration:
    2026-01-01 00:00:01,001 [learner.py] =>   seed: 1993
    2026-01-01 00:00:01,001 [learner.py] =>   model_merge_incremental: True
    2026-01-01 00:00:02,000 [learner.py] => [Training] Task 0
    2026-01-01 00:00:03,000 [learner.py] => [Evaluation] Task 0: Acc=90.00, FAA=90.00, FFM=0.00, ASA=90.00
    2026-01-01 00:00:03,000 [learner.py] => [Training] Task 1
    2026-01-01 00:00:04,000 [learner.py] => [Evaluation] Task 1: Acc=80.00, FAA=85.00, FFM=5.00, ASA=87.50
    2026-01-01 00:00:04,000 [learner.py] => [Evaluation] Accuracy matrix:
    2026-01-01 00:00:04,000 [learner.py] =>   [90.00]
    2026-01-01 00:00:04,000 [learner.py] =>   [85.00, 80.00]
    2026-01-01 00:00:04,000 [learner.py] => [Evaluation] Worst-task retention: 0.94
    2026-01-01 00:00:05,000 [learner.py] => [Summary] ╔══ END-OF-RUN RESOURCE REPORT ══╗
    2026-01-01 00:00:05,000 [learner.py] => [Summary]   Method : kcea_ablation__model_merge_incremental_True  |  Dataset : vtab  |  Seed : 1993
    2026-01-01 00:00:06,000 [learner.py] => Starting experiment: vtab - kcea_ablation__model_merge_incremental_True - seed 1994
    2026-01-01 00:00:06,001 [learner.py] => Dataset Training Size: 1000
    2026-01-01 00:00:06,001 [learner.py] => Configuration:
    2026-01-01 00:00:06,001 [learner.py] =>   seed: 1994
    2026-01-01 00:00:06,001 [learner.py] =>   model_merge_incremental: True
    2026-01-01 00:00:07,000 [learner.py] => [Training] Task 0
    2026-01-01 00:00:08,000 [learner.py] => [Evaluation] Task 0: Acc=92.00, FAA=92.00, FFM=0.00, ASA=92.00
    2026-01-01 00:00:08,000 [learner.py] => [Training] Task 1
    2026-01-01 00:00:09,000 [learner.py] => [Evaluation] Task 1: Acc=82.00, FAA=87.00, FFM=4.00, ASA=89.00
    2026-01-01 00:00:09,000 [learner.py] => [Evaluation] Accuracy matrix:
    2026-01-01 00:00:09,000 [learner.py] =>   [92.00]
    2026-01-01 00:00:09,000 [learner.py] =>   [87.00, 82.00]
    2026-01-01 00:00:09,000 [learner.py] => [Evaluation] Worst-task retention: 0.95
    2026-01-01 00:00:10,000 [learner.py] => [Summary] ╔══ END-OF-RUN RESOURCE REPORT ══╗
    2026-01-01 00:00:10,000 [learner.py] => [Summary]   Method : kcea_ablation__model_merge_incremental_True  |  Dataset : vtab  |  Seed : 1994
""")

# Single-seed file, setting value 'False' (no duplicate header).
_LOG_NON_INCREMENTAL = textwrap.dedent("""\
    2026-01-01 00:10:00,000 [learner.py] => Starting experiment: vtab - kcea_ablation__model_merge_incremental_False - seed 1993
    2026-01-01 00:10:00,001 [learner.py] => Dataset Training Size: 1000
    2026-01-01 00:10:00,001 [learner.py] => Configuration:
    2026-01-01 00:10:00,001 [learner.py] =>   seed: 1993
    2026-01-01 00:10:00,001 [learner.py] =>   model_merge_incremental: False
    2026-01-01 00:10:01,000 [learner.py] => [Training] Task 0
    2026-01-01 00:10:02,000 [learner.py] => [Evaluation] Accuracy matrix:
    2026-01-01 00:10:02,000 [learner.py] =>   [88.00]
    2026-01-01 00:10:02,000 [learner.py] =>   [83.00, 79.00]
    2026-01-01 00:10:02,001 [learner.py] => [Evaluation] Worst-task retention: 0.92
    2026-01-01 00:10:03,000 [learner.py] => [Summary] ╔══ END-OF-RUN RESOURCE REPORT ══╗
""")


@pytest.fixture()
def log_incremental(tmp_path: Path) -> Path:
    p = tmp_path / "kcea_ablation__model_merge_incremental_True.log"
    p.write_text(_LOG_INCREMENTAL)
    return p


@pytest.fixture()
def log_non_incremental(tmp_path: Path) -> Path:
    p = tmp_path / "kcea_ablation__model_merge_incremental_False.log"
    p.write_text(_LOG_NON_INCREMENTAL)
    return p


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestParseLog:
    def test_two_seeds_extracted(self, log_incremental):
        runs = parse_log(log_incremental, param="model_merge_incremental")
        assert len(runs) == 2

    def test_seed_values(self, log_incremental):
        runs = parse_log(log_incremental, param="model_merge_incremental")
        seeds = {r["seed"] for r in runs}
        assert seeds == {"1993", "1994"}

    def test_setting_extracted(self, log_incremental):
        runs = parse_log(log_incremental, param="model_merge_incremental")
        for r in runs:
            assert r["setting"] == "True"

    def test_perf_matrix_shape(self, log_incremental):
        runs = parse_log(log_incremental, param="model_merge_incremental")
        for r in runs:
            perf = r["perf"]
            assert len(perf) == 2          # 2 tasks
            assert len(perf[0]) == 1
            assert len(perf[1]) == 2

    def test_perf_values_seed1993(self, log_incremental):
        runs = parse_log(log_incremental, param="model_merge_incremental")
        run = next(r for r in runs if r["seed"] == "1993")
        assert run["perf"][0] == pytest.approx([90.0])
        assert run["perf"][1] == pytest.approx([85.0, 80.0])

    def test_no_duplicate_header_single_seed(self, log_non_incremental):
        runs = parse_log(log_non_incremental, param="model_merge_incremental")
        assert len(runs) == 1
        assert runs[0]["setting"] == "False"
        assert runs[0]["seed"] == "1993"

    def test_perf_values_non_incremental(self, log_non_incremental):
        runs = parse_log(log_non_incremental, param="model_merge_incremental")
        assert runs[0]["perf"][1] == pytest.approx([83.0, 79.0])


class TestParseLogs:
    def test_combined_two_files(self, log_incremental, log_non_incremental):
        settings = parse_logs(
            [log_incremental, log_non_incremental],
            param="model_merge_incremental",
        )
        assert set(settings.keys()) == {"True", "False"}
        assert set(settings["True"].keys()) == {"1993", "1994"}
        assert set(settings["False"].keys()) == {"1993"}

    def test_performance_key_present(self, log_incremental):
        settings = parse_logs([log_incremental], param="model_merge_incremental")
        for seed, run in settings["True"].items():
            assert "performance" in run
            assert "config" in run
