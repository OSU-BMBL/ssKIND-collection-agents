"""
Pipeline 2, Step F — LabelMergerStep

Input  : data/5.doublet/{dataset_id}.h5ad
         data/5.doublet/{dataset_id}_labels.csv      (MapMyCells output)
         data/5.doublet/{dataset_id}_annotation_result.json
Output : data/6.labeled/{dataset_id}.h5ad
         data/6.labeled/{dataset_id}_merge_result.json

Ported from reference_ignore/Mapmycell/process_labeled_h5ad.py.
Merges MapMyCells cell-type labels onto the obs DataFrame and assigns a
final `cell_type` column using confidence thresholds:
  - Human: supercluster_bootstrapping_probability >= 0.5  → supercluster_name
  - Mouse: class_bootstrapping_probability        >= 0.5  → class_name
  - Low-confidence or missing labels              → "Unknown"

If the annotation step was skipped (species not human/mouse, or tool not
available), the h5ad is still written to 6.labeled/ with cell_type = "Unknown"
so downstream steps have a consistent input path.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

import anndata as ad
import pandas as pd

logger = logging.getLogger(__name__)

DOUBLET_SUBDIR = "5.doublet"
LABELED_SUBDIR = "6.labeled"
LABELS_CSV_FILENAME = "{dataset_id}_labels.csv"
ANNOTATION_RESULT_FILENAME = "{dataset_id}_annotation_result.json"
MERGE_RESULT_FILENAME = "{dataset_id}_merge_result.json"

# Confidence thresholds (from reference)
HUMAN_PROB_THRESHOLD = 0.5
MOUSE_PROB_THRESHOLD = 0.5


class LabelMergerStep:
    """
    Pipeline 2, Step F: merge MapMyCells labels into h5ad obs.

    Input  : data/5.doublet/{dataset_id}.h5ad
             data/5.doublet/{dataset_id}_labels.csv
    Output : data/6.labeled/{dataset_id}.h5ad
             data/6.labeled/{dataset_id}_merge_result.json
    """

    def __init__(self, data_folder: Optional[str] = None) -> None:
        self.data_folder = data_folder or os.getenv("DATA_FOLDER", ".")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def merge(self, dataset_id: str) -> Optional[dict]:
        """
        Merge labels and write labeled h5ad.

        Returns a result dict, or None on unexpected failure.
        Skips if the output h5ad already exists.
        """
        out_h5ad = self._output_h5ad_path(dataset_id)
        result_path = self._result_path(dataset_id)

        if os.path.exists(out_h5ad) and os.path.exists(result_path):
            logger.info("LabelMergerStep: %s already labeled, loading from disk", dataset_id)
            with open(result_path) as f:
                return json.load(f)

        in_h5ad = self._input_h5ad_path(dataset_id)
        if not os.path.exists(in_h5ad):
            logger.error("LabelMergerStep: doublet h5ad not found for %s", dataset_id)
            return None

        try:
            adata = ad.read_h5ad(in_h5ad)
        except Exception as exc:
            logger.error("LabelMergerStep: failed to read %s: %s", in_h5ad, exc)
            return None

        os.makedirs(os.path.join(self.data_folder, LABELED_SUBDIR), exist_ok=True)

        # Determine species from obs (written by FormatConverterStep → Dataset_id lookup)
        # Fall back to checking annotation_result
        species = self._resolve_species(dataset_id, adata)

        labels_path = self._labels_path(dataset_id)
        annotation_status = self._load_annotation_status(dataset_id)

        if os.path.exists(labels_path):
            result = self._merge_with_labels(
                adata, dataset_id, labels_path, species, out_h5ad
            )
        else:
            # No labels file — assign "Unknown" for all cells
            reason = annotation_status or "no labels CSV found"
            result = self._assign_unknown(adata, dataset_id, out_h5ad, reason)

        self._write_result(result_path, result)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _merge_with_labels(
        self,
        adata: ad.AnnData,
        dataset_id: str,
        labels_path: str,
        species: str,
        out_h5ad: str,
    ) -> dict:
        try:
            labels_df = pd.read_csv(labels_path, index_col=0, comment="#")
        except Exception as exc:
            logger.error("LabelMergerStep: failed to read labels for %s: %s", dataset_id, exc)
            return self._assign_unknown(adata, dataset_id, out_h5ad, f"label CSV read error: {exc}")

        # Join labels onto obs (inner by barcode index)
        adata.obs = adata.obs.join(labels_df, how="left")

        # Fix dtypes
        for col in adata.obs.columns:
            dtype = adata.obs[col].dtype
            if dtype == "object" or dtype.name == "category":
                if "probability" in col.lower():
                    adata.obs[col] = pd.to_numeric(adata.obs[col], errors="coerce").fillna(0.0)
                elif "label" in col.lower():
                    adata.obs[col] = adata.obs[col].astype(str).fillna("")
                else:
                    adata.obs[col] = adata.obs[col].astype(str).fillna("Unknown")

        # Assign cell_type
        if "human" in species.lower():
            adata = _assign_cell_type(
                adata,
                prob_col="supercluster_bootstrapping_probability",
                name_col="supercluster_name",
                threshold=HUMAN_PROB_THRESHOLD,
            )
        elif "mouse" in species.lower():
            adata = _assign_cell_type(
                adata,
                prob_col="class_bootstrapping_probability",
                name_col="class_name",
                threshold=MOUSE_PROB_THRESHOLD,
            )
        else:
            adata.obs["cell_type"] = "Unknown"

        n_unknown = int((adata.obs["cell_type"] == "Unknown").sum())
        n_cells = adata.n_obs
        adata.uns["annotation_status"] = "annotated"
        adata.write_h5ad(out_h5ad, compression="gzip")

        logger.info(
            "LabelMergerStep: %s — %d cells, %d Unknown (%.1f%%)",
            dataset_id, n_cells, n_unknown, 100.0 * n_unknown / max(n_cells, 1),
        )
        return {
            "dataset_id": dataset_id,
            "status": "success",
            "species": species,
            "n_cells": n_cells,
            "n_unknown": n_unknown,
            "pct_unknown": round(100.0 * n_unknown / max(n_cells, 1), 1),
            "merged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _assign_unknown(
        self,
        adata: ad.AnnData,
        dataset_id: str,
        out_h5ad: str,
        reason: str,
    ) -> dict:
        adata.obs["cell_type"] = "Unknown"
        adata.uns["annotation_status"] = "no_labels"
        adata.write_h5ad(out_h5ad, compression="gzip")
        logger.info("LabelMergerStep: %s — all cells marked Unknown (%s)", dataset_id, reason)
        return {
            "dataset_id": dataset_id,
            "status": "success_no_labels",
            "message": reason,
            "n_cells": adata.n_obs,
            "n_unknown": adata.n_obs,
            "pct_unknown": 100.0,
            "merged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _resolve_species(self, dataset_id: str, adata: ad.AnnData) -> str:
        """Try to determine species from annotation_config; fall back to conversion_config."""
        annotation_config_path = os.path.join(
            self.data_folder, DOUBLET_SUBDIR,
            f"{dataset_id}_annotation_config.json",
        )
        if os.path.exists(annotation_config_path):
            with open(annotation_config_path) as f:
                cfg = json.load(f)
            taxonomy = cfg.get("taxonomy", "none")
            if "human" in taxonomy:
                return "Human"
            if "mouse" in taxonomy:
                return "Mouse"

        conv_config_path = os.path.join(
            self.data_folder, "2.raw", dataset_id, "conversion_config.json"
        )
        if os.path.exists(conv_config_path):
            with open(conv_config_path) as f:
                cfg = json.load(f)
            return cfg.get("species", "Unknown")

        return "Unknown"

    def _load_annotation_status(self, dataset_id: str) -> Optional[str]:
        path = os.path.join(
            self.data_folder, DOUBLET_SUBDIR,
            ANNOTATION_RESULT_FILENAME.format(dataset_id=dataset_id),
        )
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        status = data.get("status", "")
        msg = data.get("message", "")
        return f"{status}: {msg}" if msg else status

    def _input_h5ad_path(self, dataset_id: str) -> str:
        return os.path.join(self.data_folder, DOUBLET_SUBDIR, f"{dataset_id}.h5ad")

    def _output_h5ad_path(self, dataset_id: str) -> str:
        return os.path.join(self.data_folder, LABELED_SUBDIR, f"{dataset_id}.h5ad")

    def _labels_path(self, dataset_id: str) -> str:
        return os.path.join(
            self.data_folder, DOUBLET_SUBDIR,
            LABELS_CSV_FILENAME.format(dataset_id=dataset_id),
        )

    def _result_path(self, dataset_id: str) -> str:
        return os.path.join(
            self.data_folder, LABELED_SUBDIR,
            MERGE_RESULT_FILENAME.format(dataset_id=dataset_id),
        )

    @staticmethod
    def _write_result(path: str, result: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f, indent=2)


# ---------------------------------------------------------------------------
# Pure helper
# ---------------------------------------------------------------------------

def _assign_cell_type(
    adata: ad.AnnData,
    prob_col: str,
    name_col: str,
    threshold: float,
) -> ad.AnnData:
    """Vectorised cell_type assignment based on probability threshold."""
    if prob_col not in adata.obs.columns:
        adata.obs[prob_col] = 0.0
    if name_col not in adata.obs.columns:
        adata.obs[name_col] = "Unknown"

    adata.obs[prob_col] = pd.to_numeric(adata.obs[prob_col], errors="coerce").fillna(0.0)
    adata.obs[name_col] = adata.obs[name_col].fillna("Unknown").astype(str)

    cell_type = adata.obs[name_col].copy()
    cell_type[adata.obs[prob_col] < threshold] = "Unknown"
    adata.obs["cell_type"] = cell_type.astype(str)
    return adata
