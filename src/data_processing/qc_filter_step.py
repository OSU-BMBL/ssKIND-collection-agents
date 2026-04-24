"""
Pipeline 2, Step C3 — QCFilterStep

Input  : data/4.qc/{dataset_id}_annotated.h5ad  (QC-annotated, from CountQCStep)
         data/4.qc/{dataset_id}_thresholds.json  (from QCReviewerStep)
Output : data/4.qc/{dataset_id}.h5ad             (filtered h5ad, ready for doublet detection)
         data/4.qc/{dataset_id}_filter_result.json

Ported from reference_ignore/data_qc/Zero_matrix_qc.py (filter logic).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse

logger = logging.getLogger(__name__)

QC_SUBDIR = "4.qc"
ANNOTATED_H5AD_FILENAME_TMPL = "{dataset_id}_annotated.h5ad"
THRESHOLDS_FILENAME_TMPL = "{dataset_id}_thresholds.json"
FILTERED_H5AD_FILENAME_TMPL = "{dataset_id}.h5ad"
FILTER_RESULT_FILENAME_TMPL = "{dataset_id}_filter_result.json"


class QCFilterStep:
    """
    Pipeline 2, Step C3: apply LLM-approved thresholds to the annotated h5ad.

    Input  : data/4.qc/{dataset_id}_annotated.h5ad
             data/4.qc/{dataset_id}_thresholds.json
    Output : data/4.qc/{dataset_id}.h5ad
             data/4.qc/{dataset_id}_filter_result.json
    """

    def __init__(self, data_folder: Optional[str] = None) -> None:
        self.data_folder = data_folder or os.getenv("DATA_FOLDER", ".")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(self, dataset_id: str) -> Optional[dict]:
        """
        Apply QC thresholds and write the filtered h5ad.

        Returns a filter_result dict, or None on failure.
        Skips if the filtered h5ad already exists.
        """
        result_path = self._result_path(dataset_id)
        filtered_path = self._filtered_path(dataset_id)

        if os.path.exists(filtered_path) and os.path.exists(result_path):
            logger.info("QCFilterStep: %s already filtered, loading from disk", dataset_id)
            with open(result_path) as f:
                return json.load(f)

        thresholds_doc = self._load_thresholds(dataset_id)
        if thresholds_doc is None:
            logger.error("QCFilterStep: thresholds not found for %s", dataset_id)
            return None

        if not thresholds_doc.get("approved", True):
            reason = thresholds_doc.get("rejection_reason", "rejected by QC reviewer")
            result = {
                "dataset_id": dataset_id,
                "status": "rejected",
                "message": reason,
                "filtered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._write_result(result_path, result)
            logger.warning("QCFilterStep: %s rejected by reviewer: %s", dataset_id, reason)
            return result

        annotated_path = self._annotated_path(dataset_id)
        if not os.path.exists(annotated_path):
            logger.error("QCFilterStep: annotated h5ad not found for %s", dataset_id)
            return None

        try:
            adata = ad.read_h5ad(annotated_path)
            adata.var_names_make_unique()
        except Exception as exc:
            logger.error("QCFilterStep: failed to read %s: %s", annotated_path, exc)
            return None

        thresholds = thresholds_doc.get("thresholds", {})
        species = (thresholds_doc.get("species") or "").lower()
        n_before = adata.n_obs

        adata = _apply_thresholds(adata, thresholds, species)

        n_after = adata.n_obs
        if n_after == 0:
            result = {
                "dataset_id": dataset_id,
                "status": "failed",
                "message": "All cells removed by QC filtering",
                "n_cells_before": n_before,
                "n_cells_after": 0,
                "filtered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._write_result(result_path, result)
            logger.error("QCFilterStep: all cells removed for %s", dataset_id)
            return result

        adata.write(filtered_path, compression="gzip")
        result = {
            "dataset_id": dataset_id,
            "status": "success",
            "n_cells_before": n_before,
            "n_cells_after": n_after,
            "n_cells_removed": n_before - n_after,
            "pct_cells_kept": round(100.0 * n_after / max(n_before, 1), 1),
            "thresholds_applied": thresholds,
            "filtered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._write_result(result_path, result)
        logger.info(
            "QCFilterStep: %s — %d → %d cells (%.1f%% kept)",
            dataset_id, n_before, n_after, result["pct_cells_kept"],
        )
        return result

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _annotated_path(self, dataset_id: str) -> str:
        return os.path.join(
            self.data_folder, QC_SUBDIR,
            ANNOTATED_H5AD_FILENAME_TMPL.format(dataset_id=dataset_id),
        )

    def _thresholds_path(self, dataset_id: str) -> str:
        return os.path.join(
            self.data_folder, QC_SUBDIR,
            THRESHOLDS_FILENAME_TMPL.format(dataset_id=dataset_id),
        )

    def _filtered_path(self, dataset_id: str) -> str:
        return os.path.join(
            self.data_folder, QC_SUBDIR,
            FILTERED_H5AD_FILENAME_TMPL.format(dataset_id=dataset_id),
        )

    def _result_path(self, dataset_id: str) -> str:
        return os.path.join(
            self.data_folder, QC_SUBDIR,
            FILTER_RESULT_FILENAME_TMPL.format(dataset_id=dataset_id),
        )

    def _load_thresholds(self, dataset_id: str) -> Optional[dict]:
        path = self._thresholds_path(dataset_id)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _write_result(path: str, result: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f, indent=2)


# ---------------------------------------------------------------------------
# Pure filtering function
# ---------------------------------------------------------------------------

def _apply_thresholds(
    adata: ad.AnnData, thresholds: dict, species: str
) -> ad.AnnData:
    """Apply QC thresholds to an annotated AnnData and return the filtered copy."""
    is_hm = "human" in species or "mouse" in species

    # Gene-level: remove genes expressed in too few cells
    min_cells = thresholds.get("min_cells", 3)
    if min_cells and min_cells > 0:
        sc.pp.filter_genes(adata, min_cells=min_cells)

    # Cell-level: compose a boolean mask
    mask = np.ones(adata.n_obs, dtype=bool)

    min_genes = thresholds.get("min_genes", 200)
    if min_genes and "n_genes_by_counts" in adata.obs.columns:
        mask &= adata.obs["n_genes_by_counts"].values >= min_genes

    if is_hm:
        max_genes = thresholds.get("max_genes")
        if max_genes and "n_genes_by_counts" in adata.obs.columns:
            mask &= adata.obs["n_genes_by_counts"].values < max_genes

        min_tc = thresholds.get("min_total_counts")
        if min_tc and "total_counts" in adata.obs.columns:
            mask &= adata.obs["total_counts"].values > min_tc

        max_tc = thresholds.get("max_total_counts")
        if max_tc and "total_counts" in adata.obs.columns:
            mask &= adata.obs["total_counts"].values < max_tc

        max_mt = thresholds.get("max_pct_mt")
        if max_mt is not None and "pct_counts_mt" in adata.obs.columns:
            mask &= adata.obs["pct_counts_mt"].values < max_mt

        max_ribo = thresholds.get("max_pct_ribo")
        if max_ribo is not None and "pct_counts_ribo" in adata.obs.columns:
            mask &= adata.obs["pct_counts_ribo"].values < max_ribo
    else:
        # Other species: only max_total_counts
        max_tc = thresholds.get("max_total_counts")
        if max_tc and "total_counts" in adata.obs.columns:
            mask &= adata.obs["total_counts"].values <= max_tc

    return adata[mask].copy()
