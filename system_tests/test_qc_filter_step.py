"""
System tests for QCFilterStep (Pipeline 2, Step C3).

All tests are non-LLM and use synthetic AnnData fixtures in tmp_path.
"""

import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.data_processing.count_qc_step import _annotate_qc_metrics
from src.data_processing.qc_filter_step import QCFilterStep, _apply_thresholds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_annotated_h5ad(
    n_cells: int = 100,
    n_hk: int = 490,
    n_mt: int = 10,
    high_mt_fraction: float = 0.2,
    seed: int = 0,
) -> ad.AnnData:
    """Synthetic annotated matrix with MT genes and QC metrics.

    Normal cells: ~1000-5000 total counts, ~2% MT → pass default thresholds.
    High-MT cells (first high_mt_fraction): MT% > 20% → fail max_pct_mt=5.
    """
    rng = np.random.default_rng(seed)
    n_genes = n_hk + n_mt
    # Base counts: enough to clear min_total_counts=500 and min_genes=200
    X = rng.integers(5, 15, size=(n_cells, n_genes)).astype(float)
    # Boost high-MT cells: MT genes get very high counts → MT% >> 5%
    n_high_mt = int(n_cells * high_mt_fraction)
    X[:n_high_mt, n_hk:] += 5000

    gene_names = [f"GENE{i}" for i in range(n_hk)] + [f"MT-GENE{i}" for i in range(n_mt)]
    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs_names = [f"Cell{i}" for i in range(n_cells)]
    adata.var_names = gene_names
    adata.obs["Dataset_id"] = "TEST"
    adata.obs["Pubmed_id"] = "99999999"
    adata = _annotate_qc_metrics(adata, "human")
    return adata


def _write_annotated(tmp_path, dataset_id: str, adata: ad.AnnData) -> None:
    qc_dir = os.path.join(str(tmp_path), "4.qc")
    os.makedirs(qc_dir, exist_ok=True)
    adata.write(os.path.join(qc_dir, f"{dataset_id}_annotated.h5ad"), compression="gzip")


def _write_thresholds(
    tmp_path,
    dataset_id: str,
    approved: bool = True,
    min_genes: int = 200,
    min_cells: int = 3,
    max_genes: int = 10000,
    min_total_counts: int = 500,
    max_total_counts: int = 100000,
    max_pct_mt: float = 5.0,
    max_pct_ribo: float = 20.0,
    species: str = "Human",
    rejection_reason: str = None,
) -> None:
    qc_dir = os.path.join(str(tmp_path), "4.qc")
    os.makedirs(qc_dir, exist_ok=True)
    doc = {
        "dataset_id": dataset_id,
        "pmid": "99999999",
        "species": species,
        "reviewed_at": "2026-04-23 10:00:00",
        "approved": approved,
        "rejection_reason": rejection_reason,
        "thresholds": {
            "min_genes": min_genes,
            "min_cells": min_cells,
            "max_genes": max_genes,
            "min_total_counts": min_total_counts,
            "max_total_counts": max_total_counts,
            "max_pct_mt": max_pct_mt,
            "max_pct_ribo": max_pct_ribo,
        },
    }
    with open(os.path.join(qc_dir, f"{dataset_id}_thresholds.json"), "w") as f:
        json.dump(doc, f)


# ---------------------------------------------------------------------------
# Tests for _apply_thresholds (pure function)
# ---------------------------------------------------------------------------

def test_apply_thresholds_removes_high_mt_cells():
    adata = _make_annotated_h5ad(n_cells=100, high_mt_fraction=0.2)
    thresholds = {
        "min_genes": 1, "min_cells": 1, "max_genes": 100000,
        "min_total_counts": 0, "max_total_counts": 10_000_000,
        "max_pct_mt": 5.0, "max_pct_ribo": 100.0,
    }
    filtered = _apply_thresholds(adata, thresholds, "human")
    assert filtered.n_obs < adata.n_obs
    assert (filtered.obs["pct_counts_mt"] < 5.0).all()


def test_apply_thresholds_removes_low_count_cells():
    adata = _make_annotated_h5ad(n_cells=80)
    thresholds = {
        "min_genes": 1, "min_cells": 1, "max_genes": 100000,
        "min_total_counts": 50000,  # high threshold → most cells removed
        "max_total_counts": 10_000_000,
        "max_pct_mt": 100.0, "max_pct_ribo": 100.0,
    }
    filtered = _apply_thresholds(adata, thresholds, "human")
    assert filtered.n_obs < adata.n_obs
    if filtered.n_obs > 0:
        assert (filtered.obs["total_counts"] > 50000).all()


def test_apply_thresholds_other_species_only_max_counts():
    """For non-human/mouse, only max_total_counts is applied."""
    adata = _make_annotated_h5ad(n_cells=60)
    thresholds = {
        "min_genes": 1, "min_cells": 1,
        "max_total_counts": 1000,
    }
    filtered = _apply_thresholds(adata, thresholds, "other")
    if filtered.n_obs > 0:
        assert (filtered.obs["total_counts"] <= 1000).all()


# ---------------------------------------------------------------------------
# Tests for QCFilterStep (integration)
# ---------------------------------------------------------------------------

def test_filter_produces_h5ad_and_result(tmp_path):
    """filter() must write the filtered h5ad and result JSON."""
    dataset_id = "99999999_01"
    adata = _make_annotated_h5ad()
    _write_annotated(tmp_path, dataset_id, adata)
    _write_thresholds(tmp_path, dataset_id, max_pct_mt=5.0)

    step = QCFilterStep(data_folder=str(tmp_path))
    result = step.filter(dataset_id)

    assert result is not None
    assert result["status"] == "success"
    assert result["n_cells_after"] < result["n_cells_before"]

    filtered_path = os.path.join(str(tmp_path), "4.qc", f"{dataset_id}.h5ad")
    assert os.path.exists(filtered_path)
    reloaded = ad.read_h5ad(filtered_path)
    assert reloaded.n_obs == result["n_cells_after"]


def test_filter_idempotent(tmp_path):
    """Second call must return cached result without rewriting."""
    dataset_id = "99999999_02"
    adata = _make_annotated_h5ad()
    _write_annotated(tmp_path, dataset_id, adata)
    _write_thresholds(tmp_path, dataset_id)

    step = QCFilterStep(data_folder=str(tmp_path))
    first = step.filter(dataset_id)
    assert first is not None

    second = step.filter(dataset_id)
    assert second is not None
    assert first["n_cells_after"] == second["n_cells_after"]


def test_filter_rejected_dataset(tmp_path):
    """filter() must return status='rejected' for a reviewer-rejected dataset."""
    dataset_id = "99999999_03"
    adata = _make_annotated_h5ad()
    _write_annotated(tmp_path, dataset_id, adata)
    _write_thresholds(
        tmp_path, dataset_id,
        approved=False, rejection_reason="All zeros",
    )

    step = QCFilterStep(data_folder=str(tmp_path))
    result = step.filter(dataset_id)
    assert result is not None
    assert result["status"] == "rejected"
    assert "zero" in result["message"].lower()


def test_filter_missing_thresholds_returns_none(tmp_path):
    """filter() must return None when thresholds JSON is absent."""
    dataset_id = "99999999_04"
    step = QCFilterStep(data_folder=str(tmp_path))
    assert step.filter(dataset_id) is None


def test_filter_missing_annotated_h5ad_returns_none(tmp_path):
    """filter() must return None when annotated h5ad is absent."""
    dataset_id = "99999999_05"
    _write_thresholds(tmp_path, dataset_id)
    step = QCFilterStep(data_folder=str(tmp_path))
    assert step.filter(dataset_id) is None


def test_filter_result_has_pct_kept(tmp_path):
    """Result must include pct_cells_kept."""
    dataset_id = "99999999_06"
    adata = _make_annotated_h5ad(n_cells=80)
    _write_annotated(tmp_path, dataset_id, adata)
    _write_thresholds(tmp_path, dataset_id)

    step = QCFilterStep(data_folder=str(tmp_path))
    result = step.filter(dataset_id)
    assert result is not None
    assert "pct_cells_kept" in result
    assert 0 <= result["pct_cells_kept"] <= 100
