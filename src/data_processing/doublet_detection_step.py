"""
Pipeline 2, Step D — DoubletDetectionStep

Input  : data/4.qc/{dataset_id}.h5ad    (QC-filtered h5ad, from QCFilterStep)
Output : data/5.doublet/{dataset_id}.h5ad
         data/5.doublet/{dataset_id}_doublet_result.json

Ported from reference_ignore/data_qc/1.run_scrublet_one.py.
Adds scrublet_score / scrublet_call / predicted_doublets to obs,
then removes predicted doublets before writing.

If the dataset has fewer than MIN_CELLS_FOR_SCRUBLET cells or features,
Scrublet is skipped and all cells are kept (marked non-doublet).
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from datetime import datetime
from typing import Optional

import anndata
import numpy as np
import scanpy as sc
from scipy import sparse

warnings.filterwarnings("ignore", category=RuntimeWarning)
anndata.settings.allow_write_nullable_strings = True

logger = logging.getLogger(__name__)

QC_SUBDIR = "4.qc"
DOUBLET_SUBDIR = "5.doublet"
DOUBLET_RESULT_FILENAME_TMPL = "{dataset_id}_doublet_result.json"

MIN_CELLS_FOR_SCRUBLET = 30
DEFAULT_EXPECTED_DOUBLET_RATE = 0.06
DEFAULT_SIM_DOUBLET_RATIO = 2.0
DEFAULT_SCORE_CUTOFF = 0.3


class DoubletDetectionStep:
    """
    Pipeline 2, Step D: run Scrublet doublet detection and filter doublets.

    Input  : data/4.qc/{dataset_id}.h5ad
    Output : data/5.doublet/{dataset_id}.h5ad
             data/5.doublet/{dataset_id}_doublet_result.json
    """

    def __init__(
        self,
        data_folder: Optional[str] = None,
        expected_doublet_rate: float = DEFAULT_EXPECTED_DOUBLET_RATE,
        sim_doublet_ratio: float = DEFAULT_SIM_DOUBLET_RATIO,
        score_cutoff: float = DEFAULT_SCORE_CUTOFF,
    ) -> None:
        self.data_folder = data_folder or os.getenv("DATA_FOLDER", ".")
        self.expected_doublet_rate = expected_doublet_rate
        self.sim_doublet_ratio = sim_doublet_ratio
        self.score_cutoff = score_cutoff

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, dataset_id: str) -> Optional[dict]:
        """
        Run Scrublet and write doublet-filtered h5ad.

        Returns a result dict, or None on unexpected failure.
        Skips if the output h5ad already exists.
        """
        out_h5ad = self._output_h5ad_path(dataset_id)
        result_path = self._result_path(dataset_id)

        if os.path.exists(out_h5ad) and os.path.exists(result_path):
            logger.info("DoubletDetectionStep: %s already done, loading from disk", dataset_id)
            with open(result_path) as f:
                return json.load(f)

        in_h5ad = self._input_h5ad_path(dataset_id)
        if not os.path.exists(in_h5ad):
            logger.error("DoubletDetectionStep: QC h5ad not found for %s: %s", dataset_id, in_h5ad)
            return None

        try:
            adata = sc.read_h5ad(in_h5ad)
        except Exception as exc:
            logger.error("DoubletDetectionStep: failed to read %s: %s", in_h5ad, exc)
            return None

        os.makedirs(os.path.join(self.data_folder, DOUBLET_SUBDIR), exist_ok=True)

        n_cells, n_features = adata.shape
        if n_cells < MIN_CELLS_FOR_SCRUBLET or n_features < MIN_CELLS_FOR_SCRUBLET:
            logger.warning(
                "DoubletDetectionStep: %s too small (%d cells, %d features) — "
                "skipping Scrublet, marking all as non-doublets",
                dataset_id, n_cells, n_features,
            )
            result = self._skip_scrublet(adata, dataset_id, out_h5ad, n_cells, n_features)
        else:
            result = self._run_scrublet(adata, dataset_id, out_h5ad, n_cells)

        self._write_result(result_path, result)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_scrublet(
        self, adata: anndata.AnnData, dataset_id: str, out_h5ad: str, n_cells: int
    ) -> dict:
        import scrublet as scr
        try:
            counts = _get_counts_matrix(adata)
            scrub = scr.Scrublet(
                counts,
                expected_doublet_rate=self.expected_doublet_rate,
                sim_doublet_ratio=self.sim_doublet_ratio,
            )
            scores, auto_preds = scrub.scrub_doublets()

            adata.obs["scrublet_score"] = scores
            adata.obs["scrublet_call"] = auto_preds
            threshold = float(scrub.threshold_) if hasattr(scrub, "threshold_") and scrub.threshold_ is not None else None
            if threshold is not None:
                adata.uns["scrublet_threshold"] = threshold
            adata.obs["predicted_doublets"] = scores > self.score_cutoff

            n_doublets = int(adata.obs["predicted_doublets"].sum())
            adata_clean = adata[~adata.obs["predicted_doublets"]].copy()
            adata_clean.obs["scrublet_call"] = adata_clean.obs["scrublet_call"].astype(str)
            adata_clean.write_h5ad(out_h5ad, compression="lzf")

            logger.info(
                "DoubletDetectionStep: %s — removed %d/%d doublets (cutoff=%.2f, threshold=%.3f)",
                dataset_id, n_doublets, n_cells, self.score_cutoff, threshold or 0,
            )
            return {
                "dataset_id": dataset_id,
                "status": "success",
                "scrublet_run": True,
                "n_cells_before": n_cells,
                "n_doublets_removed": n_doublets,
                "n_cells_after": n_cells - n_doublets,
                "pct_doublets": round(100.0 * n_doublets / max(n_cells, 1), 2),
                "scrublet_threshold": threshold,
                "score_cutoff": self.score_cutoff,
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as exc:
            logger.error("DoubletDetectionStep: Scrublet failed for %s: %s", dataset_id, exc)
            # Attempt to save unfiltered h5ad so the pipeline can continue
            try:
                adata.write_h5ad(out_h5ad, compression="lzf")
            except Exception:
                pass
            return {
                "dataset_id": dataset_id,
                "status": "failed",
                "message": str(exc),
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

    @staticmethod
    def _skip_scrublet(
        adata: anndata.AnnData, dataset_id: str, out_h5ad: str, n_cells: int, n_features: int
    ) -> dict:
        adata.obs["scrublet_score"] = 0.0
        adata.obs["scrublet_call"] = "False"
        adata.obs["predicted_doublets"] = False
        adata.uns["scrublet_threshold"] = None
        adata.write_h5ad(out_h5ad, compression="lzf")
        return {
            "dataset_id": dataset_id,
            "status": "success",
            "scrublet_run": False,
            "scrublet_skip_reason": f"too small ({n_cells} cells, {n_features} features)",
            "n_cells_before": n_cells,
            "n_doublets_removed": 0,
            "n_cells_after": n_cells,
            "pct_doublets": 0.0,
            "scrublet_threshold": None,
            "score_cutoff": DEFAULT_SCORE_CUTOFF,
            "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _input_h5ad_path(self, dataset_id: str) -> str:
        return os.path.join(self.data_folder, QC_SUBDIR, f"{dataset_id}.h5ad")

    def _output_h5ad_path(self, dataset_id: str) -> str:
        return os.path.join(self.data_folder, DOUBLET_SUBDIR, f"{dataset_id}.h5ad")

    def _result_path(self, dataset_id: str) -> str:
        return os.path.join(
            self.data_folder, DOUBLET_SUBDIR,
            DOUBLET_RESULT_FILENAME_TMPL.format(dataset_id=dataset_id),
        )

    @staticmethod
    def _write_result(path: str, result: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f, indent=2)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_counts_matrix(adata: anndata.AnnData):
    """Prefer counts layer → raw → X; always return a CSR matrix."""
    if "counts" in adata.layers:
        M = adata.layers["counts"]
    elif adata.raw is not None:
        M = adata.raw.X
    else:
        M = adata.X
    return M.tocsr() if sparse.issparse(M) else sparse.csr_matrix(M)
