"""
Pipeline 2, Step B2 — FormatConverterStep

Input  : data/2.raw/{dataset_id}/conversion_config.json
         data/2.raw/{dataset_id}/{primary_file}
Output : data/3.h5ad/{dataset_id}.h5ad  (human)
       | data/3.Mh5ad/{dataset_id}.h5ad (mouse)
         data/2.raw/{dataset_id}/conversion_result.json

Ported from reference_ignore/data_precessing/h5_convert.py.
HPC hard-coded paths replaced with env vars HUMAN_GENE_MART_PATH /
MOUSE_GENE_MART_PATH.  Batch/Google-Sheet machinery removed.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

logger = logging.getLogger(__name__)

CONVERSION_CONFIG_FILENAME = "conversion_config.json"
CONVERSION_RESULT_FILENAME = "conversion_result.json"

RAW_SUBDIR = "2.raw"
HUMAN_H5AD_SUBDIR = "3.h5ad"
MOUSE_H5AD_SUBDIR = "3.Mh5ad"


# ---------------------------------------------------------------------------
# Core converter (ported from SingleCellConverter)
# ---------------------------------------------------------------------------

class SingleCellConverter:
    """Convert single-cell count files to AnnData h5ad format."""

    def read_data(self, file_path: str, format_type: str) -> ad.AnnData:
        dispatch = {
            "10x": self._read_10x,
            "10x_matrix": self._read_10x_matrix,
            "csv": self._read_delimited,
            "tsv": self._read_delimited,
            "txt": self._read_delimited,
            "csv.gz": self._read_delimited,
            "tsv.gz": self._read_delimited,
            "mtx": self._read_mtx,
            "h5": self._read_h5,
            "h5ad": self._read_h5ad,
        }
        reader = dispatch.get(format_type)
        if reader is None:
            raise ValueError(f"Unsupported format_type: {format_type!r}")
        logger.info("Reading %s as format=%s", file_path, format_type)
        return reader(file_path)

    def _read_10x(self, file_path: str) -> ad.AnnData:
        if os.path.isdir(file_path):
            adata = sc.read_10x_mtx(file_path, var_names="gene_symbols", cache=False)
        else:
            adata = self._read_10x_matrix(file_path)
        logger.info("10x: %s", adata.shape)
        return adata

    def _read_10x_matrix(self, file_path: str) -> ad.AnnData:
        prefix = os.path.basename(file_path)
        path_dir = os.path.dirname(file_path)
        adata = sc.read_10x_mtx(path_dir, prefix=prefix, var_names="gene_symbols", cache=False)
        logger.info("10x_matrix: %s", adata.shape)
        return adata

    def _read_delimited(self, file_path: str) -> ad.AnnData:
        file_lower = file_path.lower()
        compression = "gzip" if file_lower.endswith(".gz") else None
        sep = "," if (file_lower.endswith(".csv") or file_lower.endswith(".csv.gz")) else "\t"
        data = pd.read_csv(file_path, sep=sep, index_col=0, compression=compression)
        row_names = data.index.tolist()
        # Transpose if genes are rows instead of columns
        if row_names and (
            str(row_names[0]).startswith("ENSG")
            or str(row_names[0]).startswith("ENSMUSG")
            or ("ACTB" in row_names and "CD74" in row_names)
            or ("Trp53" in row_names and "Cd74" in row_names)
        ):
            data = data.T
        adata = ad.AnnData(X=data.values)
        adata.obs_names = data.index.astype(str)
        adata.var_names = data.columns.astype(str)
        logger.info("delimited: %s", adata.shape)
        return adata

    def _read_mtx(self, file_path: str) -> ad.AnnData:
        adata = sc.read_mtx(file_path)
        logger.info("mtx: %s", adata.shape)
        return adata

    def _read_h5(self, file_path: str) -> ad.AnnData:
        adata = sc.read_10x_h5(file_path, gex_only=True)
        adata.var_names_make_unique()
        logger.info("h5: %s", adata.shape)
        return adata

    def _read_h5ad(self, file_path: str) -> ad.AnnData:
        try:
            adata = ad.read_h5ad(file_path)
        except Exception:
            import h5py
            with h5py.File(file_path, "r") as f:
                for key in ("X", "counts", "data"):
                    if key in f:
                        X = f[key][:]
                        break
                else:
                    X = f[list(f.keys())[0]][:]
            adata = ad.AnnData(X=X)
        logger.info("h5ad: %s", adata.shape)
        return adata

    # ------------------------------------------------------------------
    # Gene-ID → symbol mapping
    # ------------------------------------------------------------------

    @staticmethod
    def load_gene_mappings(
        human_path: Optional[str] = None,
        mouse_path: Optional[str] = None,
    ) -> tuple[dict, dict]:
        """Return (human_mapping_dict, mouse_mapping_dict).

        Paths fall back to env vars HUMAN_GENE_MART_PATH / MOUSE_GENE_MART_PATH.
        Returns empty dicts when a path is not configured.
        """
        def _load(path: Optional[str]) -> dict:
            if not path or not os.path.exists(path):
                return {}
            df = pd.read_csv(path, sep="\t")
            return dict(zip(df["gene_ids"], df["gene_symbols"]))

        h_path = human_path or os.getenv("HUMAN_GENE_MART_PATH")
        m_path = mouse_path or os.getenv("MOUSE_GENE_MART_PATH")
        return _load(h_path), _load(m_path)

    def convert_to_symbol_counts(
        self, adata: ad.AnnData, mapping_dict: dict
    ) -> ad.AnnData:
        """Map Ensembl gene IDs to symbols; collapse duplicates by summing."""
        sample_gene = adata.var_names[0] if len(adata.var_names) else ""

        if "gene_ids" in adata.var.columns and "gene_symbols" in adata.var.columns:
            adata.var["gene_symbols"] = adata.var_names
        elif sample_gene.startswith("ENSG") or sample_gene.startswith("ENSMUSG"):
            if "gene_ids" not in adata.var.columns:
                adata.var["gene_ids"] = adata.var_names.copy()
            clean_ids = adata.var_names.str.replace(r"\.\d+$", "", regex=True)
            mapped = clean_ids.map(mapping_dict)
            missing = mapped.isnull().sum()
            if missing:
                sample_miss = adata.var["gene_ids"][mapped.isnull()].values[: min(5, missing)]
                logger.warning("%d gene IDs unmapped. Examples: %s", missing, sample_miss)
            adata.var_names = mapped.fillna(adata.var["gene_ids"])

            # Collapse duplicated gene symbols by summing
            if adata.var_names.duplicated().any():
                logger.info("Collapsing duplicate gene symbols by sum")
                X = adata.X.toarray() if sparse.issparse(adata.X) else np.array(adata.X)
                df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
                df = df.T.groupby(level=0).sum().T
                obs_copy = adata.obs.copy()
                adata = ad.AnnData(X=df.values)
                adata.obs_names = df.index
                adata.var_names = df.columns
                adata.obs = obs_copy
        else:
            # Already symbols — store reverse mapping as gene_ids
            adata.var.index.name = "gene_symbols"
            adata.var = pd.DataFrame(index=adata.var_names)
            rev = {v: k for k, v in mapping_dict.items() if pd.notnull(v)}
            clean = adata.var_names.str.replace(r"\.\d+$", "", regex=True)
            adata.var["gene_ids"] = clean.map(rev)

        adata.var["gene_symbols"] = adata.var_names
        adata.var["gene_ids"] = adata.var.get("gene_ids", pd.Series(dtype=str)).astype(str).fillna("NA_gene")
        adata.var_names = adata.var["gene_symbols"]
        adata.var_names_make_unique()
        adata.var.index = adata.var_names
        return adata

    # ------------------------------------------------------------------
    # Normalisation heuristic
    # ------------------------------------------------------------------

    @staticmethod
    def looks_logged(adata: ad.AnnData, check_n: int = 5000) -> bool:
        """Return True when values look like log1p-transformed (not raw counts)."""
        X = adata.X
        vals = X.data if sparse.issparse(X) else X.ravel()
        if vals.size == 0:
            return False
        sample = vals[: min(check_n, vals.size)]
        return bool((sample.max() <= 20.0) and not np.allclose(sample, np.floor(sample)))

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    @staticmethod
    def save_h5ad(
        adata: ad.AnnData,
        output_path: str,
        dataset_id: str,
        pmid: str,
        compression: str = "gzip",
    ) -> None:
        adata.obs = pd.DataFrame(index=adata.obs_names)
        adata.obs["Dataset_id"] = dataset_id
        adata.obs["Pubmed_id"] = pmid
        if adata.X is None:
            if "counts" in adata.layers:
                adata.X = adata.layers["counts"]
            elif adata.raw is not None:
                adata.X = adata.raw.X.copy()
        adata.raw = None
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info("Saving h5ad: %s", output_path)
        adata.write(output_path, compression=compression)

    # ------------------------------------------------------------------
    # Single-file entry point
    # ------------------------------------------------------------------

    def convert(
        self,
        input_path: str,
        output_path: str,
        dataset_id: str,
        pmid: str,
        data_type: str,
        mapping_dict: Optional[dict] = None,
    ) -> None:
        adata = self.read_data(input_path, data_type)
        if mapping_dict:
            adata = self.convert_to_symbol_counts(adata, mapping_dict)
        adata.obs["normalization_status"] = (
            "log-transformed" if self.looks_logged(adata) else "not log-transformed"
        )
        self.save_h5ad(adata, output_path, dataset_id, pmid)


# ---------------------------------------------------------------------------
# Pipeline step wrapper
# ---------------------------------------------------------------------------

class FormatConverterStep:
    """
    Pipeline 2, Step B2: convert downloaded raw files to h5ad.

    Reads  : data/2.raw/{dataset_id}/conversion_config.json
    Writes : data/3.h5ad/{dataset_id}.h5ad  OR  data/3.Mh5ad/{dataset_id}.h5ad
             data/2.raw/{dataset_id}/conversion_result.json
    """

    def __init__(
        self,
        data_folder: Optional[str] = None,
        human_gene_mart_path: Optional[str] = None,
        mouse_gene_mart_path: Optional[str] = None,
    ) -> None:
        self.data_folder = data_folder or os.getenv("DATA_FOLDER", ".")
        self._human_mart = human_gene_mart_path
        self._mouse_mart = mouse_gene_mart_path
        self._converter = SingleCellConverter()
        self._gene_maps: Optional[tuple[dict, dict]] = None  # lazy load

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(self, dataset_id: str) -> Optional[dict]:
        """
        Convert dataset_id to h5ad.

        Returns a result dict with at minimum:
          - status: "success" | "skipped" | "requires_r_extraction" | "failed"
        Returns None only on unexpected internal errors.
        """
        result_path = self._result_path(dataset_id)
        if os.path.exists(result_path):
            with open(result_path) as f:
                cached = json.load(f)
            if cached.get("status") == "success":
                logger.info("FormatConverterStep: %s already converted, skipping", dataset_id)
                return cached

        config = self._load_config(dataset_id)
        if config is None:
            logger.error("FormatConverterStep: conversion_config.json not found for %s", dataset_id)
            return None

        if config.get("requires_r_extraction"):
            result = {
                "dataset_id": dataset_id,
                "status": "requires_r_extraction",
                "message": "File requires R/rpy2 for extraction — skipped by Python pipeline",
                "converted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._write_result(result_path, result)
            logger.info("FormatConverterStep: %s requires R extraction — flagged", dataset_id)
            return result

        species = (config.get("species") or "").lower()
        if "mouse" in species:
            h5ad_subdir = MOUSE_H5AD_SUBDIR
        elif "human" in species:
            h5ad_subdir = HUMAN_H5AD_SUBDIR
        else:
            result = {
                "dataset_id": dataset_id,
                "status": "failed",
                "message": f"Unknown species '{config.get('species')}' — cannot determine output directory",
                "converted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._write_result(result_path, result)
            logger.error("FormatConverterStep: unknown species for %s", dataset_id)
            return result

        output_path = os.path.join(self.data_folder, h5ad_subdir, f"{dataset_id}.h5ad")
        if os.path.exists(output_path):
            result = {
                "dataset_id": dataset_id,
                "status": "skipped",
                "output_path": output_path,
                "message": "h5ad already exists",
                "converted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._write_result(result_path, result)
            logger.info("FormatConverterStep: %s h5ad already exists, skipping", dataset_id)
            return result

        primary_file = config.get("primary_file", "")
        input_path = os.path.join(self.data_folder, RAW_SUBDIR, dataset_id, primary_file)
        if not os.path.exists(input_path):
            result = {
                "dataset_id": dataset_id,
                "status": "failed",
                "message": f"Primary file not found: {input_path}",
                "converted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._write_result(result_path, result)
            logger.error("FormatConverterStep: primary file missing for %s: %s", dataset_id, input_path)
            return result

        mapping_dict = self._get_mapping(config)
        try:
            self._converter.convert(
                input_path=input_path,
                output_path=output_path,
                dataset_id=dataset_id,
                pmid=config.get("pmid", ""),
                data_type=config["data_type"],
                mapping_dict=mapping_dict if config.get("gene_mapping_needed") else None,
            )
            result = {
                "dataset_id": dataset_id,
                "status": "success",
                "output_path": output_path,
                "data_type": config["data_type"],
                "species": config.get("species"),
                "converted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._write_result(result_path, result)
            logger.info(
                "FormatConverterStep: converted %s → %s", dataset_id, output_path
            )
            return result
        except Exception as exc:
            result = {
                "dataset_id": dataset_id,
                "status": "failed",
                "message": str(exc),
                "converted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._write_result(result_path, result)
            logger.error("FormatConverterStep: conversion failed for %s: %s", dataset_id, exc)
            return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _config_path(self, dataset_id: str) -> str:
        return os.path.join(
            self.data_folder, RAW_SUBDIR, dataset_id, CONVERSION_CONFIG_FILENAME
        )

    def _result_path(self, dataset_id: str) -> str:
        return os.path.join(
            self.data_folder, RAW_SUBDIR, dataset_id, CONVERSION_RESULT_FILENAME
        )

    def _load_config(self, dataset_id: str) -> Optional[dict]:
        path = self._config_path(dataset_id)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _write_result(path: str, result: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f, indent=2)

    def _get_mapping(self, config: dict) -> Optional[dict]:
        """Return the appropriate gene-ID→symbol mapping dict for this config."""
        if not config.get("gene_mapping_needed"):
            return None
        if self._gene_maps is None:
            self._gene_maps = SingleCellConverter.load_gene_mappings(
                self._human_mart, self._mouse_mart
            )
        human_map, mouse_map = self._gene_maps
        species = (config.get("species") or "").lower()
        if "mouse" in species:
            return mouse_map or None
        return human_map or None
