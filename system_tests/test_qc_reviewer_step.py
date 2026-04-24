"""
System tests for QCReviewerStep (Pipeline 2, Step C2).

Non-LLM tests verify idempotency, missing-file handling, and the
threshold-building logic using a mock LLM.
LLM tests are skipped by default — remove @pytest.mark.skip() to run.
"""

import json
import logging
import os
from unittest.mock import MagicMock

import pytest

from src.agents.qc_reviewer_step import (
    QCReviewerStep,
    QCThresholdsResult,
)

logger = logging.getLogger(__name__)

FIXTURE_ROOT = "system_tests/data"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_qc_report(
    dataset_id: str = "TEST_01",
    species: str = "Human",
    n_cells: int = 500,
    mean_pct_mt: float = 3.5,
) -> dict:
    return {
        "dataset_id": dataset_id,
        "pmid": "99999999",
        "species": species,
        "computed_at": "2026-04-23 10:00:00",
        "summary": {
            "n_cells": n_cells,
            "n_genes": 2000,
            "total_counts": {"mean": 5000.0, "median": 4800.0, "p5": 600.0, "p95": 12000.0, "min": 100.0, "max": 50000.0},
            "n_genes_by_counts": {"mean": 1500.0, "median": 1400.0, "p5": 300.0, "p95": 4000.0, "min": 50.0, "max": 9500.0},
            "mean_pct_mt": mean_pct_mt,
            "mean_pct_ribo": 8.2,
            "pct_mt": {"mean": mean_pct_mt, "median": 3.0, "p5": 0.5, "p95": 9.0, "min": 0.0, "max": 25.0},
            "pct_ribo": {"mean": 8.2, "median": 7.5, "p5": 1.0, "p95": 20.0, "min": 0.0, "max": 45.0},
        },
        "suggested_thresholds": {
            "min_genes": 200,
            "min_cells": 3,
            "max_genes": 10000,
            "min_total_counts": 500,
            "max_total_counts": 100000,
            "max_pct_mt": 5.0,
            "max_pct_ribo": 20.0,
        },
    }


def _write_qc_report(tmp_path, dataset_id: str, report: dict) -> None:
    qc_dir = os.path.join(str(tmp_path), "4.qc")
    os.makedirs(qc_dir, exist_ok=True)
    with open(os.path.join(qc_dir, f"{dataset_id}_qc_report.json"), "w") as f:
        json.dump(report, f)


def _mock_llm_result(**kwargs) -> QCThresholdsResult:
    defaults = dict(
        reasoning_process="Data looks clean; keeping defaults.",
        approved=True,
        rejection_reason=None,
        min_genes=200,
        min_cells=3,
        max_genes=10000,
        min_total_counts=500,
        max_total_counts=100000,
        max_pct_mt=5.0,
        max_pct_ribo=20.0,
        notes=None,
    )
    defaults.update(kwargs)
    return QCThresholdsResult(**defaults)


# ---------------------------------------------------------------------------
# Non-LLM tests
# ---------------------------------------------------------------------------

def test_missing_report_returns_none(tmp_path):
    """review() must return None when the QC report is absent."""
    step = QCReviewerStep(llm=MagicMock(), data_folder=str(tmp_path))
    assert step.review("NO_DATASET") is None


def test_review_idempotent(tmp_path):
    """Second call must load from disk without invoking the LLM again."""
    dataset_id = "TEST_01"
    report = _make_qc_report(dataset_id)
    _write_qc_report(tmp_path, dataset_id, report)

    mock_llm = MagicMock()
    mock_result = _mock_llm_result()
    step = QCReviewerStep(llm=mock_llm, data_folder=str(tmp_path))

    # Patch _run_llm to return a canned result
    step._run_llm = MagicMock(return_value=mock_result)
    first = step.review(dataset_id)
    assert first is not None

    # Reset mock; second call should NOT invoke _run_llm
    step._run_llm.reset_mock()
    second = step.review(dataset_id)
    assert second is not None
    step._run_llm.assert_not_called()
    assert first["thresholds"] == second["thresholds"]


def test_build_thresholds_structure():
    """_build_thresholds must produce all required keys."""
    report = _make_qc_report()
    result = _mock_llm_result()
    thresholds = QCReviewerStep._build_thresholds("TEST_01", report, result)

    required = {"dataset_id", "pmid", "species", "reviewed_at", "reasoning",
                "approved", "rejection_reason", "thresholds", "notes"}
    assert required <= thresholds.keys()
    assert thresholds["approved"] is True
    t = thresholds["thresholds"]
    assert t["min_genes"] == 200
    assert t["max_pct_mt"] == 5.0


def test_build_thresholds_rejected():
    """_build_thresholds should propagate approved=False and rejection_reason."""
    report = _make_qc_report()
    result = _mock_llm_result(
        approved=False,
        rejection_reason="All cells have zero counts",
    )
    thresholds = QCReviewerStep._build_thresholds("TEST_01", report, result)
    assert thresholds["approved"] is False
    assert "zero" in (thresholds["rejection_reason"] or "").lower()


def test_review_writes_thresholds_json(tmp_path):
    """review() must write thresholds JSON to 4.qc/."""
    dataset_id = "TEST_02"
    report = _make_qc_report(dataset_id)
    _write_qc_report(tmp_path, dataset_id, report)

    step = QCReviewerStep(llm=MagicMock(), data_folder=str(tmp_path))
    step._run_llm = MagicMock(return_value=_mock_llm_result())
    result = step.review(dataset_id)

    assert result is not None
    thresh_file = os.path.join(str(tmp_path), "4.qc", f"{dataset_id}_thresholds.json")
    assert os.path.exists(thresh_file)
    with open(thresh_file) as f:
        saved = json.load(f)
    assert saved["approved"] is True
    assert saved["thresholds"]["min_genes"] == 200


# ---------------------------------------------------------------------------
# LLM tests (skipped by default)
# ---------------------------------------------------------------------------

@pytest.mark.skip()
@pytest.mark.parametrize("dataset_id, species", [
    ("39578645_01", "Human"),
    ("39578645_03", "Mouse"),
])
def test_reviewer_produces_valid_thresholds(llm, tmp_path, dataset_id, species):
    """LLM must return approved thresholds with all required fields."""
    # Expects a QC report fixture at system_tests/data/4.qc/{dataset_id}_qc_report.json
    import shutil
    qc_dir = os.path.join(str(tmp_path), "4.qc")
    os.makedirs(qc_dir, exist_ok=True)
    shutil.copy(
        os.path.join(FIXTURE_ROOT, "4.qc", f"{dataset_id}_qc_report.json"),
        os.path.join(qc_dir, f"{dataset_id}_qc_report.json"),
    )
    step = QCReviewerStep(llm=llm, data_folder=str(tmp_path))
    thresholds = step.review(dataset_id)

    assert thresholds is not None
    assert "approved" in thresholds
    t = thresholds["thresholds"]
    assert t["min_genes"] >= 200
    assert t["min_cells"] >= 3
    assert t["max_total_counts"] > 0

    logger.info(
        "%s → approved=%s thresholds=%s",
        dataset_id, thresholds["approved"], t,
    )
