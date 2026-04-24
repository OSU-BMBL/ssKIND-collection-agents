"""
System tests for CountQCStep (Pipeline 2, Step C1).

All tests are non-LLM and use synthetic AnnData written to tmp_path.
No network calls, no real data files needed.
"""

import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.data_processing.count_qc_step import (
    CountQCStep,
    _annotate_qc_metrics,
    _build_qc_report,
    DEFAULT_THRESHOLDS_HM,
    DEFAULT_THRESHOLDS_OTHER,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_h5ad(n_cells: int = 50, n_genes: int = 20, seed: int = 0) -> ad.AnnData:
    """Synthetic integer count matrix."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 500, size=(n_cells, n_genes)).astype(float)
    gene_names = [f"Gene{i}" for i in range(n_genes)]
    cell_names = [f"Cell{i}" for i in range(n_cells)]
    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs_names = cell_names
    adata.var_names = gene_names
    adata.obs["Dataset_id"] = "TEST"
    adata.obs["Pubmed_id"] = "99999999"
    return adata


def _make_h5ad_with_mt(n_cells: int = 60, n_hk: int = 18, n_mt: int = 2) -> ad.AnnData:
    """Matrix with some MT- genes for metric validation."""
    rng = np.random.default_rng(42)
    n_genes = n_hk + n_mt
    X = rng.integers(0, 300, size=(n_cells, n_genes)).astype(float)
    # Make MT genes have relatively high counts
    X[:, n_hk:] += 500
    gene_names = [f"GENE{i}" for i in range(n_hk)] + [f"MT-GENE{i}" for i in range(n_mt)]
    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs_names = [f"Cell{i}" for i in range(n_cells)]
    adata.var_names = gene_names
    return adata


def _write_conversion_config(tmp_path, dataset_id: str, species: str = "Human") -> None:
    raw_dir = os.path.join(str(tmp_path), "2.raw", dataset_id)
    os.makedirs(raw_dir, exist_ok=True)
    config = {
        "dataset_id": dataset_id,
        "pmid": "99999999",
        "analyzed_at": "2026-04-23 10:00:00",
        "data_type": "csv",
        "primary_file": "counts.csv",
        "species": species,
        "gene_mapping_needed": False,
        "normalization_status": "raw_counts",
        "requires_r_extraction": False,
        "special_handling": None,
    }
    with open(os.path.join(raw_dir, "conversion_config.json"), "w") as f:
        json.dump(config, f)


def _write_h5ad(tmp_path, dataset_id: str, adata: ad.AnnData, species: str = "Human") -> str:
    subdir = "3.Mh5ad" if "mouse" in species.lower() else "3.h5ad"
    out_dir = os.path.join(str(tmp_path), subdir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{dataset_id}.h5ad")
    adata.write(path, compression="gzip")
    return path


# ---------------------------------------------------------------------------
# Tests for pure helper functions
# ---------------------------------------------------------------------------

def test_annotate_qc_metrics_adds_columns():
    """_annotate_qc_metrics must add total_counts and n_genes_by_counts."""
    adata = _make_h5ad()
    adata = _annotate_qc_metrics(adata, "human")
    assert "total_counts" in adata.obs.columns
    assert "n_genes_by_counts" in adata.obs.columns
    assert "pct_counts_mt" in adata.obs.columns
    assert "pct_counts_ribo" in adata.obs.columns


def test_annotate_qc_mt_genes_detected():
    """MT- genes should be flagged and their counts reflected in pct_counts_mt."""
    adata = _make_h5ad_with_mt()
    adata = _annotate_qc_metrics(adata, "human")
    # MT genes have boosted counts → mean MT% should be > 0
    assert adata.obs["pct_counts_mt"].mean() > 0


def test_build_qc_report_structure():
    """_build_qc_report must return all required top-level keys."""
    adata = _make_h5ad()
    adata = _annotate_qc_metrics(adata, "human")
    config = {"pmid": "12345", "species": "Human"}
    report = _build_qc_report("TEST_01", config, adata, "human")

    required_keys = {"dataset_id", "pmid", "species", "computed_at", "summary", "suggested_thresholds"}
    assert required_keys <= report.keys()
    assert report["summary"]["n_cells"] == 50
    assert report["summary"]["n_genes"] == 20
    assert "total_counts" in report["summary"]
    assert "n_genes_by_counts" in report["summary"]
    # human → MT / ribo stats present
    assert "mean_pct_mt" in report["summary"]
    assert "pct_ribo" in report["summary"]


def test_build_qc_report_other_species_omits_mt():
    """For other species, MT/ribo stats should be absent from the report."""
    adata = _make_h5ad()
    adata = _annotate_qc_metrics(adata, "other")
    config = {"pmid": "12345", "species": "Other"}
    report = _build_qc_report("TEST_02", config, adata, "other")
    assert "mean_pct_mt" not in report["summary"]
    assert report["suggested_thresholds"] == DEFAULT_THRESHOLDS_OTHER


def test_build_qc_report_human_uses_hm_thresholds():
    adata = _make_h5ad()
    adata = _annotate_qc_metrics(adata, "human")
    config = {"pmid": "12345", "species": "Human"}
    report = _build_qc_report("TEST_03", config, adata, "human")
    assert report["suggested_thresholds"] == DEFAULT_THRESHOLDS_HM


# ---------------------------------------------------------------------------
# Tests for CountQCStep (integration)
# ---------------------------------------------------------------------------

def test_run_produces_report_and_annotated_h5ad(tmp_path):
    """run() must create both the JSON report and the annotated h5ad."""
    dataset_id = "99999999_01"
    adata = _make_h5ad_with_mt()
    _write_conversion_config(tmp_path, dataset_id, species="Human")
    _write_h5ad(tmp_path, dataset_id, adata, species="Human")

    step = CountQCStep(data_folder=str(tmp_path))
    report = step.run(dataset_id)

    assert report is not None, "run() returned None"
    assert report["summary"]["n_cells"] == adata.n_obs

    # Annotated h5ad on disk
    annotated = os.path.join(str(tmp_path), "4.qc", f"{dataset_id}_annotated.h5ad")
    assert os.path.exists(annotated)
    reloaded = ad.read_h5ad(annotated)
    assert "total_counts" in reloaded.obs.columns
    assert "pct_counts_mt" in reloaded.obs.columns

    # JSON report on disk
    report_file = os.path.join(str(tmp_path), "4.qc", f"{dataset_id}_qc_report.json")
    assert os.path.exists(report_file)


def test_run_idempotent(tmp_path):
    """Second run() call must return the same report without recomputing."""
    dataset_id = "99999999_02"
    adata = _make_h5ad()
    _write_conversion_config(tmp_path, dataset_id, species="Human")
    _write_h5ad(tmp_path, dataset_id, adata, species="Human")

    step = CountQCStep(data_folder=str(tmp_path))
    first = step.run(dataset_id)
    second = step.run(dataset_id)

    assert first is not None
    assert second is not None
    assert first["summary"]["n_cells"] == second["summary"]["n_cells"]


def test_run_mouse_uses_mouse_h5ad_path(tmp_path):
    """For Mouse species, run() must look in 3.Mh5ad/."""
    dataset_id = "99999999_03"
    adata = _make_h5ad()
    _write_conversion_config(tmp_path, dataset_id, species="Mouse")
    _write_h5ad(tmp_path, dataset_id, adata, species="Mouse")

    step = CountQCStep(data_folder=str(tmp_path))
    report = step.run(dataset_id)
    assert report is not None
    assert report["species"] == "Mouse"


def test_run_missing_config_returns_none(tmp_path):
    """run() must return None when conversion_config.json is absent."""
    step = CountQCStep(data_folder=str(tmp_path))
    assert step.run("no_such_dataset") is None


def test_run_missing_h5ad_returns_none(tmp_path):
    """run() must return None when the h5ad file doesn't exist."""
    dataset_id = "99999999_04"
    _write_conversion_config(tmp_path, dataset_id, species="Human")
    # Intentionally don't write h5ad
    step = CountQCStep(data_folder=str(tmp_path))
    assert step.run(dataset_id) is None


def test_run_requires_r_extraction_skips(tmp_path):
    """run() must return None (and not crash) for datasets requiring R extraction."""
    dataset_id = "99999999_05"
    raw_dir = os.path.join(str(tmp_path), "2.raw", dataset_id)
    os.makedirs(raw_dir, exist_ok=True)
    config = {
        "dataset_id": dataset_id, "pmid": "99999999",
        "data_type": "rds", "primary_file": "big.rds.gz",
        "species": "Human", "gene_mapping_needed": False,
        "normalization_status": "unknown", "requires_r_extraction": True,
        "special_handling": None,
    }
    with open(os.path.join(raw_dir, "conversion_config.json"), "w") as f:
        json.dump(config, f)
    step = CountQCStep(data_folder=str(tmp_path))
    assert step.run(dataset_id) is None
