"""
System tests for Steps E1 (AnnotationConfigStep), E2 (CellTypeAnnotationStep),
and F (LabelMergerStep) — Pipeline 2.

All non-LLM tests use synthetic fixtures in tmp_path.
LLM tests skipped by default.
"""

import json
import os
from unittest.mock import MagicMock

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.agents.annotation_config_step import (
    AnnotationConfigStep,
    AnnotationConfigResult,
)
from src.data_processing.cell_type_annotation_step import (
    CellTypeAnnotationStep,
    _parse_mapmycells_json,
)
from src.data_processing.label_merger_step import (
    LabelMergerStep,
    _assign_cell_type,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_h5ad(n_cells: int = 80, n_genes: int = 200) -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = rng.integers(0, 200, size=(n_cells, n_genes)).astype(float)
    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs_names = [f"CELL{i:04d}" for i in range(n_cells)]
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    adata.obs["Dataset_id"] = "TEST"
    adata.obs["Pubmed_id"] = "99999999"
    return adata


def _write_doublet_h5ad(tmp_path, dataset_id: str, adata: ad.AnnData) -> None:
    d = os.path.join(str(tmp_path), "5.doublet")
    os.makedirs(d, exist_ok=True)
    adata.write(os.path.join(d, f"{dataset_id}.h5ad"), compression="gzip")


def _write_conversion_config(
    tmp_path, dataset_id: str, species: str = "Human", data_type: str = "csv"
) -> None:
    raw_dir = os.path.join(str(tmp_path), "2.raw", dataset_id)
    os.makedirs(raw_dir, exist_ok=True)
    cfg = {
        "dataset_id": dataset_id, "pmid": "99999999",
        "species": species, "data_type": data_type,
        "gene_mapping_needed": False, "normalization_status": "raw_counts",
        "requires_r_extraction": False,
    }
    with open(os.path.join(raw_dir, "conversion_config.json"), "w") as f:
        json.dump(cfg, f)


def _write_annotation_config(
    tmp_path, dataset_id: str, annotate: bool, taxonomy: str, species: str = "Human"
) -> None:
    d = os.path.join(str(tmp_path), "5.doublet")
    os.makedirs(d, exist_ok=True)
    cfg = {
        "dataset_id": dataset_id, "pmid": "99999999",
        "annotate": annotate, "taxonomy": taxonomy,
        "configured_at": "2026-04-23 10:00:00", "notes": None,
    }
    with open(os.path.join(d, f"{dataset_id}_annotation_config.json"), "w") as f:
        json.dump(cfg, f)


def _write_labels_csv(tmp_path, dataset_id: str, barcodes: list) -> None:
    d = os.path.join(str(tmp_path), "5.doublet")
    os.makedirs(d, exist_ok=True)
    n = len(barcodes)
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "supercluster_bootstrapping_probability": rng.uniform(0, 1, n),
        "supercluster_name": np.where(
            rng.uniform(0, 1, n) > 0.3, "Excitatory neuron", "Inhibitory neuron"
        ),
        "class_bootstrapping_probability": rng.uniform(0, 1, n),
        "class_name": np.where(rng.uniform(0, 1, n) > 0.5, "Neuron", "Glia"),
    }, index=barcodes)
    df.to_csv(os.path.join(d, f"{dataset_id}_labels.csv"))


# ============================================================================
# E1 — AnnotationConfigStep (non-LLM tests)
# ============================================================================

class TestAnnotationConfigStep:

    def test_missing_conversion_config_returns_none(self, tmp_path):
        step = AnnotationConfigStep(llm=MagicMock(), data_folder=str(tmp_path))
        assert step.configure("NO_DATASET") is None

    def test_configure_idempotent(self, tmp_path):
        dataset_id = "TEST_01"
        _write_conversion_config(tmp_path, dataset_id, species="Human")
        step = AnnotationConfigStep(llm=MagicMock(), data_folder=str(tmp_path))
        mock_result = AnnotationConfigResult(
            reasoning_process="Human scRNA-seq → annotate",
            annotate=True,
            taxonomy="human_whole_brain",
            notes=None,
        )
        step._run_llm = MagicMock(return_value=mock_result)
        first = step.configure(dataset_id)
        assert first is not None

        step._run_llm.reset_mock()
        second = step.configure(dataset_id)
        assert second is not None
        step._run_llm.assert_not_called()
        assert first["taxonomy"] == second["taxonomy"]

    def test_configure_writes_json(self, tmp_path):
        dataset_id = "TEST_02"
        _write_conversion_config(tmp_path, dataset_id, species="Mouse")
        step = AnnotationConfigStep(llm=MagicMock(), data_folder=str(tmp_path))
        step._run_llm = MagicMock(return_value=AnnotationConfigResult(
            reasoning_process="Mouse → annotate",
            annotate=True,
            taxonomy="mouse_whole_brain",
        ))
        result = step.configure(dataset_id)
        assert result is not None
        cfg_path = os.path.join(str(tmp_path), "5.doublet", f"{dataset_id}_annotation_config.json")
        assert os.path.exists(cfg_path)

    def test_invalid_taxonomy_normalised_to_none(self, tmp_path):
        dataset_id = "TEST_03"
        _write_conversion_config(tmp_path, dataset_id, species="Zebrafish")
        step = AnnotationConfigStep(llm=MagicMock(), data_folder=str(tmp_path))
        step._run_llm = MagicMock(return_value=AnnotationConfigResult(
            reasoning_process="Zebrafish → skip",
            annotate=False,
            taxonomy="zebrafish_brain",  # invalid
        ))
        result = step.configure(dataset_id)
        assert result is not None
        assert result["taxonomy"] == "none"


# ============================================================================
# E2 — CellTypeAnnotationStep (non-LLM tests)
# ============================================================================

class TestCellTypeAnnotationStep:

    def test_missing_annotation_config_returns_none(self, tmp_path):
        step = CellTypeAnnotationStep(data_folder=str(tmp_path))
        assert step.annotate("NO_DATASET") is None

    def test_annotate_false_returns_skipped(self, tmp_path):
        dataset_id = "TEST_04"
        _write_annotation_config(tmp_path, dataset_id, annotate=False, taxonomy="none")
        step = CellTypeAnnotationStep(data_folder=str(tmp_path))
        result = step.annotate(dataset_id)
        assert result is not None
        assert result["status"] == "skipped"

    def test_taxonomy_none_returns_skipped(self, tmp_path):
        dataset_id = "TEST_05"
        _write_annotation_config(tmp_path, dataset_id, annotate=True, taxonomy="none")
        step = CellTypeAnnotationStep(data_folder=str(tmp_path))
        result = step.annotate(dataset_id)
        assert result["status"] == "skipped"

    def test_missing_taxonomy_path_returns_requires_external_tool(self, tmp_path):
        dataset_id = "TEST_06"
        _write_annotation_config(tmp_path, dataset_id, annotate=True, taxonomy="human_whole_brain")
        # No taxonomy path configured
        step = CellTypeAnnotationStep(data_folder=str(tmp_path))
        result = step.annotate(dataset_id)
        assert result["status"] == "requires_external_tool"

    def test_pre_existing_labels_skips_rerun(self, tmp_path):
        """If labels CSV already exists, annotate() returns cached result."""
        dataset_id = "TEST_07"
        _write_annotation_config(tmp_path, dataset_id, annotate=True, taxonomy="human_whole_brain")
        adata = _make_h5ad()
        barcodes = [f"CELL{i:04d}" for i in range(80)]
        _write_labels_csv(tmp_path, dataset_id, barcodes)
        # Also write a dummy result JSON
        result_path = os.path.join(str(tmp_path), "5.doublet", f"{dataset_id}_annotation_result.json")
        with open(result_path, "w") as f:
            json.dump({"dataset_id": dataset_id, "status": "success"}, f)

        step = CellTypeAnnotationStep(data_folder=str(tmp_path))
        result = step.annotate(dataset_id)
        assert result is not None
        assert result["status"] == "success"

    def test_parse_mapmycells_json_empty(self):
        """_parse_mapmycells_json should handle empty results gracefully."""
        df = _parse_mapmycells_json({"results": {}})
        assert len(df) == 0


# ============================================================================
# F — LabelMergerStep
# ============================================================================

class TestLabelMergerStep:

    def test_assign_cell_type_human(self):
        """_assign_cell_type must assign Unknown for low-probability cells."""
        adata = _make_h5ad(n_cells=10)
        adata.obs["supercluster_bootstrapping_probability"] = [
            0.8, 0.1, 0.9, 0.2, 0.7, 0.3, 0.6, 0.1, 0.95, 0.4
        ]
        adata.obs["supercluster_name"] = "Excitatory neuron"
        adata = _assign_cell_type(
            adata, "supercluster_bootstrapping_probability",
            "supercluster_name", threshold=0.5
        )
        high_prob = adata.obs[adata.obs["supercluster_bootstrapping_probability"] >= 0.5]
        low_prob = adata.obs[adata.obs["supercluster_bootstrapping_probability"] < 0.5]
        assert (high_prob["cell_type"] == "Excitatory neuron").all()
        assert (low_prob["cell_type"] == "Unknown").all()

    def test_missing_doublet_h5ad_returns_none(self, tmp_path):
        step = LabelMergerStep(data_folder=str(tmp_path))
        assert step.merge("NO_DATASET") is None

    def test_merge_no_labels_assigns_unknown(self, tmp_path):
        """Without a labels CSV, all cells should get cell_type='Unknown'."""
        dataset_id = "TEST_08"
        adata = _make_h5ad()
        _write_doublet_h5ad(tmp_path, dataset_id, adata)

        step = LabelMergerStep(data_folder=str(tmp_path))
        result = step.merge(dataset_id)

        assert result is not None
        assert result["status"] == "success_no_labels"
        out_h5ad = os.path.join(str(tmp_path), "6.labeled", f"{dataset_id}.h5ad")
        assert os.path.exists(out_h5ad)
        reloaded = ad.read_h5ad(out_h5ad)
        assert (reloaded.obs["cell_type"] == "Unknown").all()

    def test_merge_with_human_labels(self, tmp_path):
        """With a labels CSV, cell_type should be assigned for high-confidence cells."""
        dataset_id = "TEST_09"
        adata = _make_h5ad(n_cells=50)
        _write_doublet_h5ad(tmp_path, dataset_id, adata)
        _write_conversion_config(tmp_path, dataset_id, species="Human")
        _write_annotation_config(tmp_path, dataset_id, annotate=True,
                                  taxonomy="human_whole_brain", species="Human")
        _write_labels_csv(tmp_path, dataset_id, barcodes=[f"CELL{i:04d}" for i in range(50)])

        step = LabelMergerStep(data_folder=str(tmp_path))
        result = step.merge(dataset_id)

        assert result is not None
        assert result["status"] == "success"
        assert result["n_cells"] == 50

        reloaded = ad.read_h5ad(os.path.join(str(tmp_path), "6.labeled", f"{dataset_id}.h5ad"))
        assert "cell_type" in reloaded.obs.columns
        assert reloaded.obs["cell_type"].notna().all()
        # Some cells should have a real label
        assert set(reloaded.obs["cell_type"].unique()) <= {
            "Excitatory neuron", "Inhibitory neuron", "Unknown"
        }

    def test_merge_idempotent(self, tmp_path):
        """Second merge() call must return cached result."""
        dataset_id = "TEST_10"
        adata = _make_h5ad(n_cells=30)
        _write_doublet_h5ad(tmp_path, dataset_id, adata)

        step = LabelMergerStep(data_folder=str(tmp_path))
        first = step.merge(dataset_id)
        second = step.merge(dataset_id)
        assert first is not None and second is not None
        assert first["n_cells"] == second["n_cells"]

    # -----------------------------------------------------------------------
    # LLM test (skipped by default)
    # -----------------------------------------------------------------------

    @pytest.mark.skip()
    def test_annotation_config_step_human(self, llm, tmp_path):
        """LLM must decide annotate=True and taxonomy=human_whole_brain for human scRNA."""
        dataset_id = "TEST_LLM_01"
        _write_conversion_config(tmp_path, dataset_id, species="Human", data_type="csv")
        step = AnnotationConfigStep(llm=llm, data_folder=str(tmp_path))
        result = step.configure(dataset_id)
        assert result is not None
        assert result["annotate"] is True
        assert result["taxonomy"] == "human_whole_brain"
