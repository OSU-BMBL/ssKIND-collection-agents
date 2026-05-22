"""
System tests for ProcessingWorkflow (Pipeline 2 orchestrator).

These exercise the LangGraph routing — short-circuit on failure / stop
conditions, full-success path, metadata fan-out, and the atlas merge hand-off —
without any LLM calls or network I/O. Each step instance's public method is
replaced with a fake that returns a canned result dict.
"""

import json
import os
from unittest.mock import MagicMock

import pytest

from src.workflow.processing_workflow import ProcessingWorkflow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wf(tmp_path) -> ProcessingWorkflow:
    wf = ProcessingWorkflow(llm=MagicMock(), data_folder=str(tmp_path))
    wf.compile()
    # Bypass metadata-file lookup with a canned dataset entry.
    wf._dataset_entry = lambda pmid, did: {"dataset_id": did, "accession_ids": ["GSE1"]}
    return wf


def _wire_steps(wf: ProcessingWorkflow, **overrides) -> None:
    """Replace each step's public method with a fake. Defaults = full success."""
    defaults = {
        "manifest": {"files": [{"filename": "a.csv"}], "raw_data_available": True},
        "download": {"files": [{"status": "success"}], "all_success": True},
        "format_analyze": {"data_type": "csv", "requires_r_extraction": False},
        "convert": {"status": "success"},
        "count_qc": {"n_cells": 100, "status": "success"},
        "qc_review": {"min_genes": 200},
        "qc_filter": {"status": "success", "n_cells_after": 90},
        "doublet": {"status": "success"},
        "annotation_config": {"annotate": True, "taxonomy": "human"},
        "cell_type_annotate": {"status": "success"},
        "label_merge": {"status": "success"},
        "atlas_clean": {"status": "success"},
    }
    defaults.update(overrides)

    wf.repository_analyst.analyze = lambda dataset: defaults["manifest"]
    wf.data_downloader.download = lambda did: defaults["download"]
    wf.format_analyzer.analyze = lambda did: defaults["format_analyze"]
    wf.format_converter.convert = lambda did: defaults["convert"]
    wf.count_qc.run = lambda did: defaults["count_qc"]
    wf.qc_reviewer.review = lambda did: defaults["qc_review"]
    wf.qc_filter.filter = lambda did: defaults["qc_filter"]
    wf.doublet.run = lambda did: defaults["doublet"]
    wf.annotation_config.configure = lambda did: defaults["annotation_config"]
    wf.cell_type_annotation.annotate = lambda did: defaults["cell_type_annotate"]
    wf.label_merger.merge = lambda did: defaults["label_merge"]
    wf.atlas_cleaner.clean = lambda did: defaults["atlas_clean"]


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------

def test_full_success_reaches_atlas_clean(tmp_path):
    wf = _make_wf(tmp_path)
    _wire_steps(wf)

    summary = wf.process_dataset("39578645", "39578645_01")

    assert summary["status"] == "completed"
    assert summary["stopped_at"] is None
    # All 12 chain nodes ran.
    assert set(summary["results"].keys()) == set(ProcessingWorkflow._CHAIN)
    assert summary["results"]["atlas_clean"]["status"] == "success"


def test_requires_r_extraction_stops_at_convert(tmp_path):
    wf = _make_wf(tmp_path)
    _wire_steps(wf, convert={"status": "requires_r_extraction"})

    summary = wf.process_dataset("39578645", "39578645_01")

    assert summary["status"] == "stopped"
    assert summary["stopped_at"] == "convert"
    # Downstream nodes never ran.
    assert "count_qc" not in summary["results"]
    assert "doublet" not in summary["results"]


def test_empty_manifest_stops_at_manifest(tmp_path):
    wf = _make_wf(tmp_path)
    _wire_steps(wf, manifest={"files": [], "raw_data_available": False})

    summary = wf.process_dataset("39578645", "39578645_01")

    assert summary["status"] == "stopped"
    assert summary["stopped_at"] == "manifest"
    assert list(summary["results"].keys()) == ["manifest"]


def test_download_all_failed_stops_at_download(tmp_path):
    wf = _make_wf(tmp_path)
    _wire_steps(wf, download={"files": [{"status": "failed"}], "all_success": False})

    summary = wf.process_dataset("39578645", "39578645_01")

    assert summary["status"] == "stopped"
    assert summary["stopped_at"] == "download"


def test_qc_filter_removes_all_cells_stops(tmp_path):
    wf = _make_wf(tmp_path)
    _wire_steps(wf, qc_filter={"status": "rejected", "n_cells_after": 0})

    summary = wf.process_dataset("39578645", "39578645_01")

    assert summary["status"] == "stopped"
    assert summary["stopped_at"] == "qc_filter"
    assert "doublet" not in summary["results"]


def test_missing_dataset_entry_stops_at_manifest(tmp_path):
    wf = _make_wf(tmp_path)
    _wire_steps(wf)
    wf._dataset_entry = lambda pmid, did: None  # simulate metadata lookup miss

    summary = wf.process_dataset("39578645", "39578645_01")

    assert summary["status"] == "stopped"
    assert summary["stopped_at"] == "manifest"
    assert summary["results"]["manifest"]["status"] == "failed"


def test_annotation_failure_still_completes(tmp_path):
    """Annotation is best-effort: a failed annotate must not abort the chain."""
    wf = _make_wf(tmp_path)
    _wire_steps(
        wf,
        cell_type_annotate={"status": "failed", "message": "MapMyCells error"},
        label_merge={"status": "success_no_labels"},
    )

    summary = wf.process_dataset("39578645", "39578645_01")

    assert summary["status"] == "completed"
    assert summary["results"]["label_merge"]["status"] == "success_no_labels"


# ---------------------------------------------------------------------------
# Fan-out + atlas hand-off
# ---------------------------------------------------------------------------

def test_extract_metadata_returns_dataset_ids(tmp_path):
    wf = _make_wf(tmp_path)
    wf.metadata_extractor.extract = lambda pmid, title, full_text: [
        {"dataset_id": "39578645_01"},
        {"dataset_id": "39578645_02"},
    ]
    assert wf.extract_metadata("39578645", "T", "body") == ["39578645_01", "39578645_02"]


def test_run_paper_processes_every_dataset(tmp_path):
    wf = _make_wf(tmp_path)
    _wire_steps(wf)
    wf.metadata_extractor.extract = lambda pmid, title, full_text: [
        {"dataset_id": "39578645_01"},
        {"dataset_id": "39578645_02"},
    ]

    result = wf.run_paper("39578645", "T", "body")

    assert result["dataset_ids"] == ["39578645_01", "39578645_02"]
    assert len(result["datasets"]) == 2
    assert all(d["status"] == "completed" for d in result["datasets"])


def test_build_atlas_passes_dataset_ids(tmp_path):
    wf = _make_wf(tmp_path)
    captured = {}

    def fake_merge(dataset_ids, atlas_name="atlas"):
        captured["ids"] = dataset_ids
        captured["name"] = atlas_name
        return {"status": "success", "atlas_name": atlas_name, "n_datasets": len(dataset_ids)}

    wf.atlas_merger.merge = fake_merge

    result = wf.build_atlas(["39578645_01", "39578645_02"], atlas_name="alzheimer")

    assert result["status"] == "success"
    assert captured["ids"] == ["39578645_01", "39578645_02"]
    assert captured["name"] == "alzheimer"


# ---------------------------------------------------------------------------
# Token-usage aggregation
# ---------------------------------------------------------------------------

def test_token_usage_aggregates_across_llm_steps(tmp_path):
    """Each LLM node's last_token_usage folds into the run total."""
    wf = _make_wf(tmp_path)
    _wire_steps(wf)
    # Fakes don't reset last_token_usage, so set distinct values per LLM step.
    wf.repository_analyst.last_token_usage = {"total_tokens": 10, "prompt_tokens": 7, "completion_tokens": 3}
    wf.format_analyzer.last_token_usage = {"total_tokens": 20, "prompt_tokens": 15, "completion_tokens": 5}
    wf.qc_reviewer.last_token_usage = {"total_tokens": 30, "prompt_tokens": 25, "completion_tokens": 5}
    wf.annotation_config.last_token_usage = {"total_tokens": 40, "prompt_tokens": 30, "completion_tokens": 10}

    summary = wf.process_dataset("39578645", "39578645_01")

    assert wf.token_usage == {"total_tokens": 100, "prompt_tokens": 77, "completion_tokens": 23}
    assert summary["token_usage"] == wf.token_usage


def test_token_usage_includes_metadata_extraction(tmp_path):
    """A0 (metadata extraction) contributes to the run total too."""
    wf = _make_wf(tmp_path)
    _wire_steps(wf)
    wf.metadata_extractor.extract = lambda pmid, title, full_text: [{"dataset_id": "39578645_01"}]
    wf.metadata_extractor.last_token_usage = {"total_tokens": 5, "prompt_tokens": 4, "completion_tokens": 1}
    # Per-dataset LLM steps simulate cache hits → contribute nothing.
    for step in (wf.repository_analyst, wf.format_analyzer, wf.qc_reviewer, wf.annotation_config):
        step.last_token_usage = None

    result = wf.run_paper("39578645", "T", "body")

    assert result["token_usage"] == {"total_tokens": 5, "prompt_tokens": 4, "completion_tokens": 1}


def test_reset_token_usage_zeros_total(tmp_path):
    wf = _make_wf(tmp_path)
    wf.token_usage = {"total_tokens": 123, "prompt_tokens": 100, "completion_tokens": 23}
    wf.reset_token_usage()
    assert wf.token_usage == {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}


def test_llm_step_skip_path_resets_token_usage(tmp_path):
    """An LLM step that hits its cache must clear last_token_usage (no LLM ran)."""
    from src.agents.format_analyzer_step import (
        FormatAnalyzerStep,
        CONVERSION_CONFIG_FILENAME,
    )

    dataset_id = "39578645_01"
    raw_dir = os.path.join(str(tmp_path), "2.raw", dataset_id)
    os.makedirs(raw_dir, exist_ok=True)
    with open(os.path.join(raw_dir, CONVERSION_CONFIG_FILENAME), "w") as f:
        json.dump({"dataset_id": dataset_id, "data_type": "csv"}, f)

    step = FormatAnalyzerStep(llm=MagicMock(), data_folder=str(tmp_path))
    # Simulate a stale value from a prior dataset on the reused instance.
    step.last_token_usage = {"total_tokens": 99, "prompt_tokens": 99, "completion_tokens": 0}

    config = step.analyze(dataset_id)  # config exists → skip path, no LLM call

    assert config is not None
    assert step.last_token_usage is None
