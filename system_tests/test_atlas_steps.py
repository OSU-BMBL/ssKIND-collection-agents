"""
System tests for AtlasCleanerStep (Step G) and AtlasMergerStep (Step H).

All tests use synthetic AnnData in tmp_path — no network calls.
"""

import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.data_processing.atlas_cleaner_step import (
    AtlasCleanerStep,
    MIN_CELLS_FOR_ATLAS,
    _clean_single,
)
from src.data_processing.atlas_merger_step import AtlasMergerStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_labeled_h5ad(
    n_cells: int = 300,
    n_genes: int = 100,
    n_unknown: int = 50,
    seed: int = 0,
) -> ad.AnnData:
    """Synthetic labeled h5ad that passes AtlasCleanerStep."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 200, size=(n_cells, n_genes)).astype(np.float32)
    gene_symbols = [f"GENE{i}" for i in range(n_genes)]
    gene_ids = [f"ENSG{i:09d}" for i in range(n_genes)]
    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs_names = [f"CELL{i:04d}" for i in range(n_cells)]
    adata.var_names = gene_symbols
    adata.var["gene_symbols"] = gene_symbols
    adata.var["gene_ids"] = gene_ids
    adata.obs["Dataset_id"] = "TEST"
    adata.obs["Pubmed_id"] = "99999999"
    # Cell type labels
    cell_types = ["Excitatory neuron"] * (n_cells - n_unknown) + ["Unknown"] * n_unknown
    adata.obs["cell_type"] = cell_types
    # Add some QC columns that should be stripped
    adata.obs["scrublet_score"] = 0.0
    adata.obs["scrublet_call"] = "False"
    return adata


def _write_labeled_h5ad(tmp_path, dataset_id: str, adata: ad.AnnData) -> None:
    d = os.path.join(str(tmp_path), "6.labeled")
    os.makedirs(d, exist_ok=True)
    adata.write(os.path.join(d, f"{dataset_id}.h5ad"), compression="gzip")


def _write_atlas_clean_h5ad(tmp_path, dataset_id: str, adata: ad.AnnData) -> None:
    d = os.path.join(str(tmp_path), "7.atlas_clean")
    os.makedirs(d, exist_ok=True)
    adata.write(os.path.join(d, f"{dataset_id}.h5ad"), compression="gzip")


# ============================================================================
# G — AtlasCleanerStep
# ============================================================================

class TestAtlasCleanerStep:

    def test_clean_success(self, tmp_path):
        """clean() must produce a valid h5ad with Unknowns removed."""
        dataset_id = "TEST_01"
        adata = _make_labeled_h5ad(n_cells=400, n_unknown=80)
        _write_labeled_h5ad(tmp_path, dataset_id, adata)

        step = AtlasCleanerStep(data_folder=str(tmp_path))
        result = step.clean(dataset_id)

        assert result is not None
        assert result["status"] == "success"
        assert result["n_cells_after"] == 400 - 80

        out = os.path.join(str(tmp_path), "7.atlas_clean", f"{dataset_id}.h5ad")
        assert os.path.exists(out)
        reloaded = ad.read_h5ad(out)
        assert reloaded.n_obs == 400 - 80
        assert "cell_type" in reloaded.obs.columns
        assert (reloaded.obs["cell_type"] != "Unknown").all()
        assert "scrublet_score" not in reloaded.obs.columns
        assert reloaded.X.dtype == np.float32
        assert "gene_ids" in reloaded.var.columns

    def test_clean_too_few_cells_returns_skipped(self, tmp_path):
        """Fewer than MIN_CELLS_FOR_ATLAS cells after removing Unknown → skipped."""
        dataset_id = "TEST_02"
        n_cells = MIN_CELLS_FOR_ATLAS + 10
        adata = _make_labeled_h5ad(n_cells=n_cells, n_unknown=n_cells - 5)
        _write_labeled_h5ad(tmp_path, dataset_id, adata)

        step = AtlasCleanerStep(data_folder=str(tmp_path))
        result = step.clean(dataset_id)

        assert result is not None
        assert result["status"] == "skipped"

    def test_clean_missing_gene_ids_returns_failed(self, tmp_path):
        """Without gene_ids in var, clean() must return status='failed'."""
        dataset_id = "TEST_03"
        adata = _make_labeled_h5ad(n_cells=300)
        del adata.var["gene_ids"]
        _write_labeled_h5ad(tmp_path, dataset_id, adata)

        step = AtlasCleanerStep(data_folder=str(tmp_path))
        result = step.clean(dataset_id)
        assert result is not None
        assert result["status"] == "failed"

    def test_clean_idempotent(self, tmp_path):
        dataset_id = "TEST_04"
        adata = _make_labeled_h5ad(n_cells=300)
        _write_labeled_h5ad(tmp_path, dataset_id, adata)

        step = AtlasCleanerStep(data_folder=str(tmp_path))
        first = step.clean(dataset_id)
        second = step.clean(dataset_id)
        assert first is not None and second is not None
        assert first["n_cells_after"] == second["n_cells_after"]

    def test_clean_missing_input_returns_none(self, tmp_path):
        step = AtlasCleanerStep(data_folder=str(tmp_path))
        assert step.clean("NO_DATASET") is None

    def test_clean_single_removes_nan_prefix_genes(self, tmp_path):
        """_clean_single must drop genes with 'nan-' prefix."""
        n_cells, n_genes = 300, 10
        X = np.ones((n_cells, n_genes), dtype=np.float32)
        adata = ad.AnnData(X=sp.csr_matrix(X))
        adata.obs_names = [f"C{i}" for i in range(n_cells)]
        gene_syms = [f"GENE{i}" for i in range(9)] + ["nan-GENE_BAD"]
        adata.var_names = gene_syms
        adata.var["gene_symbols"] = gene_syms
        adata.var["gene_ids"] = [f"ENSG{i}" for i in range(n_genes)]
        adata.obs["cell_type"] = "Neuron"

        out = os.path.join(str(tmp_path), "out.h5ad")
        result = _clean_single(adata, "TEST_NAN", out)
        assert result["status"] == "success"
        reloaded = ad.read_h5ad(out)
        assert all(not g.startswith("nan-") for g in reloaded.var_names)
        assert reloaded.n_vars == 9

    def test_clean_obs_names_prefixed_with_dataset_id(self, tmp_path):
        """Each cell's obs_name should be prefixed with the dataset_id."""
        dataset_id = "DS42"
        adata = _make_labeled_h5ad(n_cells=250, n_unknown=0)
        _write_labeled_h5ad(tmp_path, dataset_id, adata)

        step = AtlasCleanerStep(data_folder=str(tmp_path))
        result = step.clean(dataset_id)
        assert result["status"] == "success"
        reloaded = ad.read_h5ad(
            os.path.join(str(tmp_path), "7.atlas_clean", f"{dataset_id}.h5ad")
        )
        assert all(obs.startswith(f"{dataset_id}_") for obs in reloaded.obs_names)


# ============================================================================
# H — AtlasMergerStep
# ============================================================================

def _make_clean_h5ad(n_cells: int = 250, n_genes: int = 80, batch_label: str = "B") -> ad.AnnData:
    """Minimal clean h5ad ready for atlas merging."""
    rng = np.random.default_rng(hash(batch_label) % 2**31)
    X = rng.integers(0, 150, size=(n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs_names = [f"{batch_label}_C{i:04d}" for i in range(n_cells)]
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    adata.var["gene_ids"] = [f"ENSG{i:09d}" for i in range(n_genes)]
    adata.obs["cell_type"] = "Neuron"
    return adata


class TestAtlasMergerStep:

    def test_merge_two_datasets(self, tmp_path):
        """Merging two clean h5ad files should produce a valid atlas."""
        ids = ["DS01", "DS02"]
        for did in ids:
            _write_atlas_clean_h5ad(tmp_path, did, _make_clean_h5ad(batch_label=did))

        step = AtlasMergerStep(
            data_folder=str(tmp_path),
            min_cells_for_gene=1,
            n_top_genes=50,
        )
        result = step.merge(ids, atlas_name="test_atlas")

        assert result is not None
        assert result["status"] == "success"
        assert result["n_datasets"] == 2
        assert result["n_cells"] == 500

        out = os.path.join(str(tmp_path), "8.atlas", "test_atlas.h5ad")
        assert os.path.exists(out)
        atlas = ad.read_h5ad(out)
        assert atlas.n_obs == 500
        assert "counts" in atlas.layers
        assert atlas.n_vars <= 50

    def test_merge_idempotent(self, tmp_path):
        ids = ["DS03", "DS04"]
        for did in ids:
            _write_atlas_clean_h5ad(tmp_path, did, _make_clean_h5ad(batch_label=did))

        step = AtlasMergerStep(data_folder=str(tmp_path), min_cells_for_gene=1, n_top_genes=50)
        first = step.merge(ids, atlas_name="atlas2")
        second = step.merge(ids, atlas_name="atlas2")
        assert first is not None and second is not None
        assert first["n_cells"] == second["n_cells"]

    def test_merge_no_clean_files_returns_failed(self, tmp_path):
        step = AtlasMergerStep(data_folder=str(tmp_path))
        result = step.merge(["NO_SUCH_DATASET"], atlas_name="empty_atlas")
        assert result is not None
        assert result["status"] == "failed"

    def test_merge_skips_missing_dataset(self, tmp_path):
        """Missing datasets should be skipped; present ones should be merged."""
        _write_atlas_clean_h5ad(tmp_path, "DS05", _make_clean_h5ad(batch_label="DS05"))
        step = AtlasMergerStep(data_folder=str(tmp_path), min_cells_for_gene=1, n_top_genes=50)
        result = step.merge(["DS05", "MISSING_DS"], atlas_name="partial_atlas")
        assert result is not None
        assert result["status"] == "success"
        assert result["n_datasets"] == 1

    def test_merge_result_has_required_keys(self, tmp_path):
        _write_atlas_clean_h5ad(tmp_path, "DS06", _make_clean_h5ad(batch_label="DS06"))
        step = AtlasMergerStep(data_folder=str(tmp_path), min_cells_for_gene=1, n_top_genes=50)
        result = step.merge(["DS06"], atlas_name="keys_atlas")
        required = {
            "atlas_name", "status", "n_datasets", "n_cells",
            "n_genes_before_hvg", "n_hvg", "merged_at",
        }
        assert result is not None
        assert required <= result.keys()
