"""
Pipeline 2, Step E2 — CellTypeAnnotationStep

Input  : data/5.doublet/{dataset_id}.h5ad
         data/5.doublet/{dataset_id}_annotation_config.json
Output : data/5.doublet/{dataset_id}_labels.csv
         data/5.doublet/{dataset_id}_annotation_result.json

Runs MapMyCells (Allen Institute hierarchical_mapping / cell_type_mapper)
if available and configured.  Falls back gracefully:
  - annotate=False in config         → status "skipped"
  - taxonomy="none"                  → status "skipped"
  - package/reference not available  → status "requires_external_tool"
  - pre-computed labels CSV present  → status "skipped" (already done)

Reference taxonomy files configured via env vars:
  HUMAN_MAPMYCELLS_TAXONOMY_PATH  — path to human taxonomy JSON
  MOUSE_MAPMYCELLS_TAXONOMY_PATH  — path to mouse taxonomy JSON

Both point to the precomputed stats directory consumed by cell_type_mapper.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

import anndata as ad

logger = logging.getLogger(__name__)

DOUBLET_SUBDIR = "5.doublet"
ANNOTATION_CONFIG_FILENAME = "{dataset_id}_annotation_config.json"
LABELS_CSV_FILENAME = "{dataset_id}_labels.csv"
ANNOTATION_RESULT_FILENAME = "{dataset_id}_annotation_result.json"


class CellTypeAnnotationStep:
    """
    Pipeline 2, Step E2: run MapMyCells cell-type annotation.

    Input  : data/5.doublet/{dataset_id}.h5ad
             data/5.doublet/{dataset_id}_annotation_config.json
    Output : data/5.doublet/{dataset_id}_labels.csv
             data/5.doublet/{dataset_id}_annotation_result.json
    """

    def __init__(
        self,
        data_folder: Optional[str] = None,
        human_taxonomy_path: Optional[str] = None,
        mouse_taxonomy_path: Optional[str] = None,
    ) -> None:
        self.data_folder = data_folder or os.getenv("DATA_FOLDER", ".")
        self._human_taxonomy = human_taxonomy_path or os.getenv("HUMAN_MAPMYCELLS_TAXONOMY_PATH")
        self._mouse_taxonomy = mouse_taxonomy_path or os.getenv("MOUSE_MAPMYCELLS_TAXONOMY_PATH")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate(self, dataset_id: str) -> Optional[dict]:
        """
        Run MapMyCells annotation for dataset_id.

        Returns a result dict, or None on unexpected failure.
        Skips if the labels CSV already exists or if annotation is disabled.
        """
        result_path = self._result_path(dataset_id)
        labels_path = self._labels_path(dataset_id)

        if os.path.exists(labels_path) and os.path.exists(result_path):
            logger.info("CellTypeAnnotationStep: %s labels already exist, loading from disk", dataset_id)
            with open(result_path) as f:
                return json.load(f)

        annotation_config = self._load_annotation_config(dataset_id)
        if annotation_config is None:
            logger.error(
                "CellTypeAnnotationStep: annotation_config not found for %s", dataset_id
            )
            return None

        if not annotation_config.get("annotate", False):
            result = self._make_result(dataset_id, "skipped",
                                       message="Annotation disabled by AnnotationConfigStep")
            self._write_result(result_path, result)
            return result

        taxonomy = annotation_config.get("taxonomy", "none")
        if taxonomy == "none":
            result = self._make_result(dataset_id, "skipped",
                                       message="taxonomy=none — skipping annotation")
            self._write_result(result_path, result)
            return result

        taxonomy_path = self._resolve_taxonomy_path(taxonomy)
        if not taxonomy_path:
            result = self._make_result(
                dataset_id, "requires_external_tool",
                message=(
                    f"Taxonomy reference file not configured for '{taxonomy}'. "
                    "Set HUMAN_MAPMYCELLS_TAXONOMY_PATH or MOUSE_MAPMYCELLS_TAXONOMY_PATH."
                ),
            )
            self._write_result(result_path, result)
            logger.warning(
                "CellTypeAnnotationStep: %s — taxonomy path not configured", dataset_id
            )
            return result

        h5ad_path = self._h5ad_path(dataset_id)
        if not os.path.exists(h5ad_path):
            logger.error("CellTypeAnnotationStep: h5ad not found for %s", dataset_id)
            return None

        return self._run_mapmycells(dataset_id, h5ad_path, taxonomy_path, labels_path, result_path)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_mapmycells(
        self,
        dataset_id: str,
        h5ad_path: str,
        taxonomy_path: str,
        labels_path: str,
        result_path: str,
    ) -> dict:
        try:
            from cell_type_mapper.cli.from_specified_markers import (
                FromSpecifiedMarkersRunner,
            )
        except ImportError:
            result = self._make_result(
                dataset_id, "requires_external_tool",
                message=(
                    "cell_type_mapper package not installed. "
                    "Install with: pip install cell-type-mapper"
                ),
            )
            self._write_result(result_path, result)
            return result

        try:
            import tempfile
            adata = ad.read_h5ad(h5ad_path)
            n_cells = adata.n_obs

            with tempfile.TemporaryDirectory() as tmp_dir:
                out_json = os.path.join(tmp_dir, "mapping.json")
                runner = FromSpecifiedMarkersRunner(
                    args=[],
                    input_path=h5ad_path,
                    precomputed_stats=taxonomy_path,
                    output_path=out_json,
                    map_to_ensembl=False,
                    log_path=os.path.join(tmp_dir, "log.txt"),
                )
                runner.run()

                with open(out_json) as f:
                    mapping = json.load(f)

            labels_df = _parse_mapmycells_json(mapping)
            os.makedirs(os.path.dirname(labels_path), exist_ok=True)
            labels_df.to_csv(labels_path)

            result = self._make_result(
                dataset_id, "success",
                n_cells=n_cells,
                taxonomy_path=taxonomy_path,
            )
            self._write_result(result_path, result)
            logger.info(
                "CellTypeAnnotationStep: %s annotated %d cells", dataset_id, n_cells
            )
            return result

        except Exception as exc:
            result = self._make_result(dataset_id, "failed", message=str(exc))
            self._write_result(result_path, result)
            logger.error("CellTypeAnnotationStep: failed for %s: %s", dataset_id, exc)
            return result

    def _resolve_taxonomy_path(self, taxonomy: str) -> Optional[str]:
        if "human" in taxonomy:
            path = self._human_taxonomy
        elif "mouse" in taxonomy:
            path = self._mouse_taxonomy
        else:
            return None
        return path if path and os.path.exists(path) else None

    def _annotation_config_path(self, dataset_id: str) -> str:
        return os.path.join(
            self.data_folder, DOUBLET_SUBDIR,
            ANNOTATION_CONFIG_FILENAME.format(dataset_id=dataset_id),
        )

    def _h5ad_path(self, dataset_id: str) -> str:
        return os.path.join(self.data_folder, DOUBLET_SUBDIR, f"{dataset_id}.h5ad")

    def _labels_path(self, dataset_id: str) -> str:
        return os.path.join(
            self.data_folder, DOUBLET_SUBDIR,
            LABELS_CSV_FILENAME.format(dataset_id=dataset_id),
        )

    def _result_path(self, dataset_id: str) -> str:
        return os.path.join(
            self.data_folder, DOUBLET_SUBDIR,
            ANNOTATION_RESULT_FILENAME.format(dataset_id=dataset_id),
        )

    def _load_annotation_config(self, dataset_id: str) -> Optional[dict]:
        path = self._annotation_config_path(dataset_id)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _make_result(
        dataset_id: str,
        status: str,
        message: str = "",
        n_cells: Optional[int] = None,
        taxonomy_path: Optional[str] = None,
    ) -> dict:
        r: dict = {
            "dataset_id": dataset_id,
            "status": status,
            "annotated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if message:
            r["message"] = message
        if n_cells is not None:
            r["n_cells"] = n_cells
        if taxonomy_path:
            r["taxonomy_path"] = taxonomy_path
        return r

    @staticmethod
    def _write_result(path: str, result: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f, indent=2)


# ---------------------------------------------------------------------------
# MapMyCells JSON → DataFrame
# ---------------------------------------------------------------------------

def _parse_mapmycells_json(mapping: dict) -> "pd.DataFrame":
    """Convert cell_type_mapper JSON output to a barcode-indexed DataFrame."""
    import pandas as pd

    rows = []
    results = mapping.get("results", {})
    for barcode, cell_data in results.items():
        row: dict = {"cell_id": barcode}
        for level, level_data in cell_data.items():
            if isinstance(level_data, dict):
                name = level_data.get("name", "")
                prob = level_data.get("bootstrapping_probability", None)
                row[f"{level}_name"] = name
                if prob is not None:
                    row[f"{level}_bootstrapping_probability"] = prob
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("cell_id")
    return df
