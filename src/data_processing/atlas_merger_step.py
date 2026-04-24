"""
Pipeline 2, Step H — AtlasMergerStep

Input  : data/7.atlas_clean/{dataset_id}.h5ad  (one per dataset)
Output : data/8.atlas/{atlas_name}.h5ad
         data/8.atlas/{atlas_name}_merge_result.json

Ported from reference_ignore/ATLAS_Human/0.2.atlas_merge.py.
Merges all cleaned single-dataset h5ad files into one atlas:
  1. Load each dataset in memory
  2. Concatenate with join="outer" (keep all genes, fill missing with 0)
  3. Filter genes expressed in < MIN_CELLS_FOR_GENE cells
  4. Store raw counts in a "counts" layer
  5. Normalize (target_sum=1e4) and log1p
  6. Select and subset to the top N highly variable genes
  7. Write final atlas

For very large collections (> batch_size datasets), intermediate batch files
are written to data/8.atlas/{atlas_name}_batches/ to avoid OOM.
"""

from __future__ import annotations

import gc
import json
import logging
import os
from datetime import datetime
from typing import List, Optional

import anndata as ad
import scanpy as sc

logger = logging.getLogger(__name__)

ATLAS_CLEAN_SUBDIR = "7.atlas_clean"
ATLAS_SUBDIR = "8.atlas"
MERGE_RESULT_FILENAME_TMPL = "{atlas_name}_merge_result.json"

DEFAULT_BATCH_SIZE = 800
DEFAULT_MIN_CELLS_FOR_GENE = 200
DEFAULT_N_TOP_GENES = 5000


class AtlasMergerStep:
    """
    Pipeline 2, Step H: merge cleaned h5ad files into a single atlas.

    Call with a list of dataset_ids to include.  If the output atlas already
    exists the step is idempotent and returns the cached result immediately.

    Input  : data/7.atlas_clean/{dataset_id}.h5ad  (for each id)
    Output : data/8.atlas/{atlas_name}.h5ad
             data/8.atlas/{atlas_name}_merge_result.json
    """

    def __init__(
        self,
        data_folder: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        min_cells_for_gene: int = DEFAULT_MIN_CELLS_FOR_GENE,
        n_top_genes: int = DEFAULT_N_TOP_GENES,
    ) -> None:
        self.data_folder = data_folder or os.getenv("DATA_FOLDER", ".")
        self.batch_size = batch_size
        self.min_cells_for_gene = min_cells_for_gene
        self.n_top_genes = n_top_genes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def merge(
        self,
        dataset_ids: List[str],
        atlas_name: str = "atlas",
    ) -> Optional[dict]:
        """
        Merge dataset_ids into a single atlas h5ad.

        Returns a result dict, or None on unexpected failure.
        Skips if the output already exists.
        """
        out_h5ad = self._output_path(atlas_name)
        result_path = self._result_path(atlas_name)

        if os.path.exists(out_h5ad) and os.path.exists(result_path):
            logger.info("AtlasMergerStep: atlas '%s' already exists, loading result", atlas_name)
            with open(result_path) as f:
                return json.load(f)

        os.makedirs(os.path.join(self.data_folder, ATLAS_SUBDIR), exist_ok=True)

        # Resolve which datasets have cleaned h5ad
        available = [d for d in dataset_ids if os.path.exists(self._clean_path(d))]
        missing = [d for d in dataset_ids if d not in available]
        if missing:
            logger.warning(
                "AtlasMergerStep: %d dataset(s) not in atlas_clean, skipping: %s",
                len(missing), missing[:5],
            )
        if not available:
            result = {
                "atlas_name": atlas_name,
                "status": "failed",
                "message": "No cleaned h5ad files found — run AtlasCleanerStep first",
                "merged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._write_result(result_path, result)
            return result

        try:
            adata = self._load_and_concat(available, atlas_name)
            n_cells_raw = adata.n_obs
            n_genes_raw = adata.n_vars

            sc.pp.filter_genes(adata, min_cells=self.min_cells_for_gene)
            adata.layers["counts"] = adata.X.copy()
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=self.n_top_genes,
                batch_key="batch",
                subset=True,
            )

            adata.write(out_h5ad, compression="gzip")
            n_hvg = adata.n_vars
            del adata
            gc.collect()

            result = {
                "atlas_name": atlas_name,
                "status": "success",
                "n_datasets": len(available),
                "n_cells": n_cells_raw,
                "n_genes_before_hvg": n_genes_raw,
                "n_hvg": n_hvg,
                "min_cells_for_gene": self.min_cells_for_gene,
                "n_top_genes": self.n_top_genes,
                "merged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._write_result(result_path, result)
            logger.info(
                "AtlasMergerStep: '%s' — %d datasets, %d cells, %d HVGs",
                atlas_name, len(available), n_cells_raw, n_hvg,
            )
            return result

        except Exception as exc:
            result = {
                "atlas_name": atlas_name,
                "status": "failed",
                "message": str(exc),
                "merged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._write_result(result_path, result)
            logger.error("AtlasMergerStep: merge failed for '%s': %s", atlas_name, exc)
            return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_and_concat(self, dataset_ids: List[str], atlas_name: str) -> ad.AnnData:
        """Load all cleaned h5ad files and concatenate, batching if necessary."""
        if len(dataset_ids) <= self.batch_size:
            return self._concat_batch(dataset_ids)

        # For large collections, process in batches and merge the batch results
        batch_dir = os.path.join(
            self.data_folder, ATLAS_SUBDIR, f"{atlas_name}_batches"
        )
        os.makedirs(batch_dir, exist_ok=True)
        batch_files = []
        for i in range(0, len(dataset_ids), self.batch_size):
            batch_ids = dataset_ids[i: i + self.batch_size]
            batch_path = os.path.join(batch_dir, f"batch_{i // self.batch_size + 1:03d}.h5ad")
            if not os.path.exists(batch_path):
                batch_adata = self._concat_batch(batch_ids)
                batch_adata.write(batch_path, compression="gzip")
                del batch_adata
                gc.collect()
            batch_files.append(batch_path)
            logger.info("AtlasMergerStep: saved batch %d", i // self.batch_size + 1)

        # Merge batch files
        batch_list = [sc.read_h5ad(f) for f in batch_files]
        combined = ad.concat(batch_list, axis=0, join="outer", index_unique=None)
        del batch_list
        gc.collect()
        return combined

    def _concat_batch(self, dataset_ids: List[str]) -> ad.AnnData:
        adata_list = []
        for did in dataset_ids:
            try:
                adata_list.append(sc.read_h5ad(self._clean_path(did)))
            except Exception as exc:
                logger.warning("AtlasMergerStep: failed to load %s: %s", did, exc)
        if not adata_list:
            raise ValueError("No datasets could be loaded in this batch")
        return ad.concat(
            adata_list,
            axis=0,
            join="outer",
            label="batch",
            keys=dataset_ids[: len(adata_list)],
            index_unique=None,
        )

    def _clean_path(self, dataset_id: str) -> str:
        return os.path.join(self.data_folder, ATLAS_CLEAN_SUBDIR, f"{dataset_id}.h5ad")

    def _output_path(self, atlas_name: str) -> str:
        return os.path.join(self.data_folder, ATLAS_SUBDIR, f"{atlas_name}.h5ad")

    def _result_path(self, atlas_name: str) -> str:
        return os.path.join(
            self.data_folder, ATLAS_SUBDIR,
            MERGE_RESULT_FILENAME_TMPL.format(atlas_name=atlas_name),
        )

    @staticmethod
    def _write_result(path: str, result: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
