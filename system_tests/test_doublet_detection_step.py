"""
System tests for DoubletDetectionStep (Pipeline 2, Step D).

All tests are non-LLM and use synthetic AnnData written to tmp_path.
"""

import json
import os

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from src.data_processing.doublet_detection_step import (
    DoubletDetectionStep,
    MIN_CELLS_FOR_SCRUBLET,
    _get_counts_matrix,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_qc_h5ad(n_cells: int = 200, n_genes: int = 500, seed: int = 0) -> ad.AnnData:
    """Synthetic integer count matrix suitable as QC-filtered input."""
    rng = np.random.default_rng(seed)
    X = rng.negative_binomial(5, 0.5, size=(n_cells, n_genes)).astype(float)
    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs_names = [f"Cell{i}" for i in range(n_cells)]
    adata.var_names = [f"Gene{i}" for i in range(n_genes)]
    adata.obs["Dataset_id"] = "TEST"
    adata.obs["Pubmed_id"] = "99999999"
    return adata


def _write_qc_h5ad(tmp_path, dataset_id: str, adata: ad.AnnData) -> str:
    qc_dir = os.path.join(str(tmp_path), "4.qc")
    os.makedirs(qc_dir, exist_ok=True)
    path = os.path.join(qc_dir, f"{dataset_id}.h5ad")
    adata.write(path, compression="gzip")
    return path


# ---------------------------------------------------------------------------
# Unit tests for helper
# ---------------------------------------------------------------------------

def test_get_counts_matrix_from_X():
    adata = _make_qc_h5ad(50, 50)
    M = _get_counts_matrix(adata)
    assert M.shape == (50, 50)
    assert sp.issparse(M)


def test_get_counts_matrix_prefers_layer():
    adata = _make_qc_h5ad(50, 50)
    layer = sp.csr_matrix(np.ones((50, 50)))
    adata.layers["counts"] = layer
    M = _get_counts_matrix(adata)
    assert M.sum() == 50 * 50  # all ones from the layer


# ---------------------------------------------------------------------------
# Integration tests for DoubletDetectionStep
# ---------------------------------------------------------------------------

def test_small_dataset_skips_scrublet(tmp_path):
    """Datasets smaller than MIN_CELLS_FOR_SCRUBLET must be kept intact."""
    dataset_id = "99999999_01"
    n_cells = MIN_CELLS_FOR_SCRUBLET - 1
    adata = _make_qc_h5ad(n_cells=n_cells, n_genes=20)
    _write_qc_h5ad(tmp_path, dataset_id, adata)

    step = DoubletDetectionStep(data_folder=str(tmp_path))
    result = step.run(dataset_id)

    assert result is not None
    assert result["status"] == "success"
    assert result["scrublet_run"] is False
    assert result["n_cells_after"] == n_cells
    assert result["n_doublets_removed"] == 0

    out_h5ad = os.path.join(str(tmp_path), "5.doublet", f"{dataset_id}.h5ad")
    assert os.path.exists(out_h5ad)
    reloaded = ad.read_h5ad(out_h5ad)
    assert reloaded.n_obs == n_cells
    assert "scrublet_score" in reloaded.obs.columns


def test_normal_dataset_runs_scrublet(tmp_path):
    """Datasets above the threshold should run Scrublet and write filtered h5ad."""
    dataset_id = "99999999_02"
    adata = _make_qc_h5ad(n_cells=300, n_genes=500)
    _write_qc_h5ad(tmp_path, dataset_id, adata)

    step = DoubletDetectionStep(data_folder=str(tmp_path))
    result = step.run(dataset_id)

    assert result is not None
    assert result["status"] == "success"
    assert result["scrublet_run"] is True
    assert result["n_cells_before"] == 300
    assert result["n_cells_after"] <= 300
    assert 0.0 <= result["pct_doublets"] <= 100.0

    out_h5ad = os.path.join(str(tmp_path), "5.doublet", f"{dataset_id}.h5ad")
    assert os.path.exists(out_h5ad)
    reloaded = ad.read_h5ad(out_h5ad)
    assert reloaded.n_obs == result["n_cells_after"]
    assert "scrublet_score" in reloaded.obs.columns


def test_run_writes_result_json(tmp_path):
    """run() must write a doublet_result.json alongside the h5ad."""
    dataset_id = "99999999_03"
    adata = _make_qc_h5ad(n_cells=150, n_genes=300)
    _write_qc_h5ad(tmp_path, dataset_id, adata)

    step = DoubletDetectionStep(data_folder=str(tmp_path))
    result = step.run(dataset_id)
    assert result is not None

    result_file = os.path.join(str(tmp_path), "5.doublet", f"{dataset_id}_doublet_result.json")
    assert os.path.exists(result_file)
    with open(result_file) as f:
        saved = json.load(f)
    assert saved["dataset_id"] == dataset_id
    assert "n_cells_after" in saved


def test_run_idempotent(tmp_path):
    """Second call must return cached result without re-running Scrublet."""
    dataset_id = "99999999_04"
    adata = _make_qc_h5ad(n_cells=200, n_genes=400)
    _write_qc_h5ad(tmp_path, dataset_id, adata)

    step = DoubletDetectionStep(data_folder=str(tmp_path))
    first = step.run(dataset_id)
    assert first is not None

    second = step.run(dataset_id)
    assert second is not None
    assert first["n_cells_after"] == second["n_cells_after"]


def test_run_missing_input_returns_none(tmp_path):
    """run() must return None when the QC h5ad is absent."""
    step = DoubletDetectionStep(data_folder=str(tmp_path))
    assert step.run("no_such_dataset") is None


def test_high_cutoff_keeps_all_cells(tmp_path):
    """With cutoff=1.0 no cell should be predicted as doublet."""
    dataset_id = "99999999_05"
    adata = _make_qc_h5ad(n_cells=200, n_genes=400)
    _write_qc_h5ad(tmp_path, dataset_id, adata)

    step = DoubletDetectionStep(data_folder=str(tmp_path), score_cutoff=1.0)
    result = step.run(dataset_id)

    assert result is not None
    assert result["status"] == "success"
    assert result["n_doublets_removed"] == 0
    assert result["n_cells_after"] == 200


def test_result_contains_required_keys(tmp_path):
    """Result dict must contain the standard set of keys."""
    dataset_id = "99999999_06"
    adata = _make_qc_h5ad(n_cells=10, n_genes=10)  # tiny → skip path
    _write_qc_h5ad(tmp_path, dataset_id, adata)

    step = DoubletDetectionStep(data_folder=str(tmp_path))
    result = step.run(dataset_id)

    required = {
        "dataset_id", "status", "scrublet_run",
        "n_cells_before", "n_doublets_removed", "n_cells_after",
        "pct_doublets", "score_cutoff", "processed_at",
    }
    assert required <= result.keys()
