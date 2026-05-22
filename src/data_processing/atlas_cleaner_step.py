"""
Pipeline 2, Step G — AtlasCleanerStep

Input  : data/6.labeled/{dataset_id}.h5ad
Output : data/7.atlas_clean/{dataset_id}.h5ad
         data/7.atlas_clean/{dataset_id}_clean_result.json

Ported from reference_ignore/ATLAS_Human/0.1_clean_atlas_unknown.py.
Prepares a single labeled h5ad for atlas inclusion:
  - validates gene_ids / gene_symbols columns
  - removes genes with invalid names ("nan-" prefix)
  - removes "Unknown" cell-type cells (only when annotation ran; skipped when annotation was disabled)
  - enforces minimum cell count (MIN_CELLS_FOR_ATLAS = 200)
  - makes obs_names globally unique: {dataset_id}_{barcode}
  - strips bulk QC/scrublet obs columns and ancillary var columns
  - converts X to float32 CSR
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

import anndata as ad
import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

LABELED_SUBDIR = "6.labeled"
ATLAS_CLEAN_SUBDIR = "7.atlas_clean"
CLEAN_RESULT_FILENAME_TMPL = "{dataset_id}_clean_result.json"

MIN_CELLS_FOR_ATLAS = 200

_DROP_OBS_COLS = {
    "log1p_n_genes_by_counts", "scrublet_call", "scrublet_score",
    "predicted_doublets", "log1p_total_counts",
    "pct_counts_in_top_50_genes", "pct_counts_in_top_100_genes",
    "pct_counts_in_top_200_genes", "pct_counts_in_top_500_genes",
    "total_counts_mt", "log1p_total_counts_mt",
    "total_counts_ribo", "log1p_total_counts_ribo",
    "total_counts_hb", "log1p_total_counts_hb", "pct_counts_hb",
}


class AtlasCleanerStep:
    """
    Pipeline 2, Step G: prepare a single labeled h5ad for atlas inclusion.

    Input  : data/6.labeled/{dataset_id}.h5ad
    Output : data/7.atlas_clean/{dataset_id}.h5ad
             data/7.atlas_clean/{dataset_id}_clean_result.json
    """

    def __init__(self, data_folder: Optional[str] = None) -> None:
        self.data_folder = data_folder or os.getenv("DATA_FOLDER", ".")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, dataset_id: str) -> Optional[dict]:
        """
        Clean a labeled h5ad for atlas inclusion.

        Returns a result dict, or None on unexpected failure.
        Skips if the output already exists.
        """
        out_h5ad = self._output_path(dataset_id)
        result_path = self._result_path(dataset_id)

        if os.path.exists(out_h5ad) and os.path.exists(result_path):
            logger.info("AtlasCleanerStep: %s already cleaned, loading from disk", dataset_id)
            with open(result_path) as f:
                return json.load(f)

        in_h5ad = self._input_path(dataset_id)
        if not os.path.exists(in_h5ad):
            logger.error("AtlasCleanerStep: labeled h5ad not found for %s", dataset_id)
            return None

        try:
            adata = ad.read_h5ad(in_h5ad)
        except Exception as exc:
            logger.error("AtlasCleanerStep: failed to read %s: %s", in_h5ad, exc)
            return None

        os.makedirs(os.path.join(self.data_folder, ATLAS_CLEAN_SUBDIR), exist_ok=True)
        result = _clean_single(adata, dataset_id, out_h5ad)
        self._write_result(result_path, result)
        return result

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def _input_path(self, dataset_id: str) -> str:
        return os.path.join(self.data_folder, LABELED_SUBDIR, f"{dataset_id}.h5ad")

    def _output_path(self, dataset_id: str) -> str:
        return os.path.join(self.data_folder, ATLAS_CLEAN_SUBDIR, f"{dataset_id}.h5ad")

    def _result_path(self, dataset_id: str) -> str:
        return os.path.join(
            self.data_folder, ATLAS_CLEAN_SUBDIR,
            CLEAN_RESULT_FILENAME_TMPL.format(dataset_id=dataset_id),
        )

    @staticmethod
    def _write_result(path: str, result: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f, indent=2)


# ---------------------------------------------------------------------------
# Pure cleaning logic (easy to unit-test independently)
# ---------------------------------------------------------------------------

def _clean_single(adata: ad.AnnData, dataset_id: str, out_path: str) -> dict:
    """Apply all cleaning transformations; write to out_path; return result dict."""
    n_cells_in = adata.n_obs

    # 1. Validate gene_ids column
    if "gene_ids" not in adata.var.columns:
        return {
            "dataset_id": dataset_id,
            "status": "failed",
            "message": "gene_ids column missing from var — run FormatConverterStep first",
            "cleaned_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    # 2. Ensure gene_symbols column exists
    if "gene_symbols" not in adata.var.columns:
        adata.var["gene_symbols"] = adata.var_names

    adata.var["gene_ids"] = adata.var["gene_ids"].astype(str)

    # 3. Replace invalid gene_ids with gene_symbols
    invalid_mask = adata.var["gene_ids"].isin({"nan", "NaN", "None", "", "NA_gene"}) | \
                   adata.var["gene_ids"].isna()
    adata.var.loc[invalid_mask, "gene_ids"] = adata.var.loc[invalid_mask, "gene_symbols"]

    # 4. Drop genes with "nan-" prefix in symbols
    valid_genes = ~adata.var["gene_symbols"].str.startswith("nan-")
    adata = adata[:, valid_genes].copy()

    # 5. Set var_names to gene_symbols, make unique
    adata.var_names = adata.var["gene_symbols"].astype(str)
    adata.var_names_make_unique()

    # 6. Remove Unknown cell-type cells — only when annotation actually ran.
    # LabelMergerStep writes uns["annotation_status"] = "annotated" when labels
    # were merged, and "no_labels" when annotation was disabled or unavailable.
    # Defaulting to "annotated" is the safe choice for h5ads written before this
    # flag existed (they were produced with annotation enabled).
    annotation_status = adata.uns.get("annotation_status", "annotated")
    if "cell_type" in adata.obs.columns and annotation_status == "annotated":
        keep = adata.obs["cell_type"] != "Unknown"
        adata = adata[keep].copy()
    elif annotation_status != "annotated":
        logger.info(
            "AtlasCleanerStep: %s — annotation was not run (status=%r), "
            "keeping all %d cells",
            dataset_id, annotation_status, adata.n_obs,
        )

    n_cells_after_unknown = adata.n_obs

    # 7. Minimum cell check
    if n_cells_after_unknown < MIN_CELLS_FOR_ATLAS:
        return {
            "dataset_id": dataset_id,
            "status": "skipped",
            "message": (
                f"Only {n_cells_after_unknown} cells remaining "
                f"(minimum {MIN_CELLS_FOR_ATLAS} required for atlas)"
            ),
            "n_cells_in": n_cells_in,
            "n_cells_after": n_cells_after_unknown,
            "cleaned_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    # 8. Make obs_names globally unique
    adata.obs_names = [f"{dataset_id}_{bc}" for bc in adata.obs_names]

    # 9. Drop QC/scrublet obs columns
    drop_cols = [c for c in _DROP_OBS_COLS if c in adata.obs.columns]
    adata.obs.drop(columns=drop_cols, inplace=True, errors="ignore")

    # 10. Keep only gene_ids in var; clear ancillary slots
    adata.var = adata.var[["gene_ids"]].copy()
    adata.layers.clear()
    adata.raw = None
    adata.obsm.clear()
    adata.varm.clear()
    adata.obsp.clear()
    adata.uns.clear()

    # 11. Convert X to float32 CSR
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    if adata.X.dtype != np.float32:
        adata.X = adata.X.astype(np.float32)

    adata.write(out_path, compression="gzip")
    logger.info(
        "AtlasCleanerStep: %s — %d → %d cells written to atlas_clean",
        dataset_id, n_cells_in, n_cells_after_unknown,
    )
    return {
        "dataset_id": dataset_id,
        "status": "success",
        "n_cells_in": n_cells_in,
        "n_cells_after": n_cells_after_unknown,
        "n_unknown_removed": n_cells_in - n_cells_after_unknown,
        "cleaned_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
