"""
Pipeline 2, Step C1 — CountQCStep

Input  : data/3.h5ad/{dataset_id}.h5ad   (human)
       | data/3.Mh5ad/{dataset_id}.h5ad  (mouse)
         data/2.raw/{dataset_id}/conversion_config.json  (species)
Output : data/4.qc/{dataset_id}_annotated.h5ad   (h5ad with QC obs columns)
         data/4.qc/{dataset_id}_qc_report.json    (summary stats + suggested thresholds)

Ported from reference_ignore/data_qc/Zero_matrix_qc.py.
Computes MT% / ribo% / count metrics but does NOT filter — that is Step C3,
after Step C2 (LLM) has reviewed and approved the thresholds.
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

RAW_SUBDIR = "2.raw"
HUMAN_H5AD_SUBDIR = "3.h5ad"
MOUSE_H5AD_SUBDIR = "3.Mh5ad"
QC_SUBDIR = "4.qc"

CONVERSION_CONFIG_FILENAME = "conversion_config.json"
CONVERSION_RESULT_FILENAME = "conversion_result.json"
QC_REPORT_FILENAME_TMPL = "{dataset_id}_qc_report.json"
ANNOTATED_H5AD_FILENAME_TMPL = "{dataset_id}_annotated.h5ad"

# Default thresholds (human / mouse)
DEFAULT_THRESHOLDS_HM = {
    "min_genes": 200,
    "min_cells": 3,
    "max_genes": 10000,
    "min_total_counts": 500,
    "max_total_counts": 100000,
    "max_pct_mt": 5.0,
    "max_pct_ribo": 20.0,
}

# Default thresholds (other species — simpler)
DEFAULT_THRESHOLDS_OTHER = {
    "min_genes": 200,
    "min_cells": 3,
    "max_total_counts": 10000,
}


class CountQCStep:
    """
    Pipeline 2, Step C1: compute QC metrics and write summary report.

    Does NOT filter; filtering is Step C3 after LLM threshold review (C2).
    """

    def __init__(self, data_folder: Optional[str] = None) -> None:
        self.data_folder = data_folder or os.getenv("DATA_FOLDER", ".")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, dataset_id: str) -> Optional[dict]:
        """
        Compute QC metrics for dataset_id.

        Returns the qc_report dict, or None on failure.
        Skips computation if the report already exists.
        """
        report_path = self._report_path(dataset_id)
        annotated_path = self._annotated_path(dataset_id)

        if os.path.exists(report_path) and os.path.exists(annotated_path):
            logger.info("CountQCStep: %s already has QC report, loading from disk", dataset_id)
            with open(report_path) as f:
                return json.load(f)

        config = self._load_config(dataset_id)
        if config is None:
            logger.error("CountQCStep: conversion_config.json not found for %s", dataset_id)
            return None

        if config.get("requires_r_extraction"):
            logger.warning(
                "CountQCStep: %s requires R extraction — skipping QC", dataset_id
            )
            return None

        h5ad_path = self._resolve_h5ad_path(dataset_id, config)
        if h5ad_path is None or not os.path.exists(h5ad_path):
            logger.error("CountQCStep: h5ad not found for %s (expected: %s)", dataset_id, h5ad_path)
            return None

        species = (config.get("species") or "").lower()
        try:
            adata = sc.read_h5ad(h5ad_path)
            adata.var_names_make_unique()
        except Exception as exc:
            logger.error("CountQCStep: failed to read %s: %s", h5ad_path, exc)
            return None

        if adata.n_obs == 0 or adata.n_vars == 0:
            logger.error("CountQCStep: empty dataset for %s (%s)", dataset_id, adata.shape)
            return None

        adata = _annotate_qc_metrics(adata, species)
        os.makedirs(os.path.join(self.data_folder, QC_SUBDIR), exist_ok=True)
        adata.write(annotated_path, compression="gzip")
        logger.info("CountQCStep: wrote annotated h5ad %s", annotated_path)

        report = _build_qc_report(dataset_id, config, adata, species)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(
            "CountQCStep: %s → %d cells, %d genes, mean_mt=%.2f%%",
            dataset_id,
            adata.n_obs,
            adata.n_vars,
            report["summary"]["mean_pct_mt"] if report["summary"].get("mean_pct_mt") is not None else 0,
        )
        return report

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _config_path(self, dataset_id: str) -> str:
        return os.path.join(self.data_folder, RAW_SUBDIR, dataset_id, CONVERSION_CONFIG_FILENAME)

    def _report_path(self, dataset_id: str) -> str:
        return os.path.join(
            self.data_folder, QC_SUBDIR,
            QC_REPORT_FILENAME_TMPL.format(dataset_id=dataset_id),
        )

    def _annotated_path(self, dataset_id: str) -> str:
        return os.path.join(
            self.data_folder, QC_SUBDIR,
            ANNOTATED_H5AD_FILENAME_TMPL.format(dataset_id=dataset_id),
        )

    def _load_config(self, dataset_id: str) -> Optional[dict]:
        path = self._config_path(dataset_id)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def _resolve_h5ad_path(self, dataset_id: str, config: dict) -> Optional[str]:
        species = (config.get("species") or "").lower()
        if "mouse" in species:
            subdir = MOUSE_H5AD_SUBDIR
        elif "human" in species:
            subdir = HUMAN_H5AD_SUBDIR
        else:
            return None
        return os.path.join(self.data_folder, subdir, f"{dataset_id}.h5ad")


# ---------------------------------------------------------------------------
# Pure functions (easy to unit-test)
# ---------------------------------------------------------------------------

def _annotate_qc_metrics(adata: ad.AnnData, species: str) -> ad.AnnData:
    """Add MT / ribo flags and compute scanpy QC metrics in-place."""
    if "gene_symbols" not in adata.var.columns:
        adata.var["gene_symbols"] = adata.var_names

    syms = adata.var["gene_symbols"].str.upper()
    adata.var["mt"] = syms.str.startswith("MT-")
    adata.var["ribo"] = syms.str.startswith(("RPS", "RPL"))

    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo"], percent_top=None, inplace=True)

    # Ensure total_counts exists for other-species path
    if "total_counts" not in adata.obs.columns:
        X = adata.X
        totals = np.asarray(X.sum(axis=1)).ravel() if scipy.sparse.issparse(X) else X.sum(axis=1)
        adata.obs["total_counts"] = totals

    return adata


def _build_qc_report(
    dataset_id: str, config: dict, adata: ad.AnnData, species: str
) -> dict:
    """Build the JSON report dict from an annotated AnnData."""
    obs = adata.obs

    def _stat(col: str) -> Optional[dict]:
        if col not in obs.columns:
            return None
        v = obs[col].dropna()
        if len(v) == 0:
            return None
        return {
            "mean": float(v.mean()),
            "median": float(v.median()),
            "p5": float(v.quantile(0.05)),
            "p95": float(v.quantile(0.95)),
            "min": float(v.min()),
            "max": float(v.max()),
        }

    is_hm = "human" in species or "mouse" in species

    summary: dict = {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "total_counts": _stat("total_counts"),
        "n_genes_by_counts": _stat("n_genes_by_counts"),
    }
    if is_hm:
        summary["mean_pct_mt"] = (
            float(obs["pct_counts_mt"].mean()) if "pct_counts_mt" in obs.columns else None
        )
        summary["mean_pct_ribo"] = (
            float(obs["pct_counts_ribo"].mean()) if "pct_counts_ribo" in obs.columns else None
        )
        summary["pct_mt"] = _stat("pct_counts_mt")
        summary["pct_ribo"] = _stat("pct_counts_ribo")

    suggested_thresholds = (
        DEFAULT_THRESHOLDS_HM.copy() if is_hm else DEFAULT_THRESHOLDS_OTHER.copy()
    )

    return {
        "dataset_id": dataset_id,
        "pmid": config.get("pmid", ""),
        "species": config.get("species", ""),
        "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
        "suggested_thresholds": suggested_thresholds,
    }
