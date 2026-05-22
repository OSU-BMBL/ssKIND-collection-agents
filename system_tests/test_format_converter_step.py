"""
System tests for FormatConverterStep (Pipeline 2, Step B2).

Non-LLM tests use synthetic in-memory fixtures (small CSV / 10x MTX) written to
tmp_path — no network calls, no real data files needed.

LLM tests are skipped by default.
"""

import gzip
import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.data_processing.format_converter import (
    CONVERSION_RESULT_FILENAME,
    FormatConverterStep,
    SingleCellConverter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_conversion_config(
    tmp_path,
    dataset_id: str,
    data_type: str,
    primary_file: str,
    species: str = "Human",
    pmid: str = "99999999",
    requires_r_extraction: bool = False,
    gene_mapping_needed: bool = False,
    normalization_status: str = "raw_counts",
) -> None:
    raw_dir = os.path.join(str(tmp_path), "2.raw", dataset_id)
    os.makedirs(raw_dir, exist_ok=True)
    config = {
        "dataset_id": dataset_id,
        "pmid": pmid,
        "analyzed_at": "2026-04-23 10:00:00",
        "data_type": data_type,
        "primary_file": primary_file,
        "species": species,
        "gene_mapping_needed": gene_mapping_needed,
        "normalization_status": normalization_status,
        "requires_r_extraction": requires_r_extraction,
        "special_handling": None,
    }
    with open(os.path.join(raw_dir, "conversion_config.json"), "w") as f:
        json.dump(config, f)


def _write_small_csv(tmp_path, dataset_id: str, filename: str, compressed: bool = False) -> None:
    """Write a tiny 3-gene × 4-cell count CSV (cells as rows, genes as columns)."""
    raw_dir = os.path.join(str(tmp_path), "2.raw", dataset_id)
    os.makedirs(raw_dir, exist_ok=True)
    data = pd.DataFrame(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=float),
        index=["cell1", "cell2", "cell3", "cell4"],
        columns=["GeneA", "GeneB", "GeneC"],
    )
    path = os.path.join(raw_dir, filename)
    if compressed:
        with gzip.open(path, "wt") as fh:
            data.to_csv(fh)
    else:
        data.to_csv(path)


def _write_10x_dir(tmp_path, dataset_id: str, dir_name: str) -> None:
    """Write a minimal 10x MTX directory (3 genes, 4 cells)."""
    raw_dir = os.path.join(str(tmp_path), "2.raw", dataset_id, dir_name)
    os.makedirs(raw_dir, exist_ok=True)
    matrix = sp.csr_matrix(
        np.array([[1, 0, 3, 0], [0, 5, 0, 7], [2, 2, 2, 2]], dtype=float)
    )
    barcodes = pd.DataFrame(["ACGT-1", "TGCA-2", "GCTA-3", "ATCG-4"])
    features = pd.DataFrame(
        [["ENSG001", "GeneA", "Gene Expression"],
         ["ENSG002", "GeneB", "Gene Expression"],
         ["ENSG003", "GeneC", "Gene Expression"]]
    )

    import scipy.io as sio
    mtx_path = os.path.join(raw_dir, "matrix.mtx")
    sio.mmwrite(mtx_path, matrix)

    # 10x v3 files are TAB-separated; features.tsv = [gene_id, symbol, type].
    with gzip.open(os.path.join(raw_dir, "barcodes.tsv.gz"), "wt") as f:
        barcodes.to_csv(f, sep="\t", index=False, header=False)
    with gzip.open(os.path.join(raw_dir, "features.tsv.gz"), "wt") as f:
        features.to_csv(f, sep="\t", index=False, header=False)
    with gzip.open(os.path.join(raw_dir, "matrix.mtx.gz"), "wb") as fout:
        with open(mtx_path, "rb") as fin:
            fout.write(fin.read())
    os.remove(mtx_path)  # keep only the gzipped matrix (10x v3 layout)


# ---------------------------------------------------------------------------
# Non-LLM unit tests
# ---------------------------------------------------------------------------

def test_requires_r_extraction_returns_flagged(tmp_path):
    """When requires_r_extraction=True, convert() must return status='requires_r_extraction'."""
    dataset_id = "99999999_01"
    _make_conversion_config(
        tmp_path, dataset_id, data_type="rds",
        primary_file="big.rds.gz", requires_r_extraction=True,
    )
    step = FormatConverterStep(data_folder=str(tmp_path))
    result = step.convert(dataset_id)

    assert result is not None
    assert result["status"] == "requires_r_extraction"

    result_file = os.path.join(str(tmp_path), "2.raw", dataset_id, CONVERSION_RESULT_FILENAME)
    assert os.path.exists(result_file)
    with open(result_file) as f:
        saved = json.load(f)
    assert saved["status"] == "requires_r_extraction"


def test_missing_config_returns_none(tmp_path):
    """convert() must return None when conversion_config.json is absent."""
    step = FormatConverterStep(data_folder=str(tmp_path))
    assert step.convert("no_such_dataset") is None


def test_missing_primary_file_returns_failed(tmp_path):
    """convert() must return status='failed' when primary file doesn't exist."""
    dataset_id = "99999999_02"
    _make_conversion_config(
        tmp_path, dataset_id, data_type="csv",
        primary_file="missing.csv", species="Human",
    )
    step = FormatConverterStep(data_folder=str(tmp_path))
    result = step.convert(dataset_id)
    assert result is not None
    assert result["status"] == "failed"
    assert "not found" in result["message"].lower() or "missing" in result["message"].lower()


def test_csv_conversion_produces_h5ad(tmp_path):
    """A small CSV file should be converted to a valid h5ad under 3.h5ad/."""
    dataset_id = "99999999_03"
    _write_small_csv(tmp_path, dataset_id, "counts.csv")
    _make_conversion_config(
        tmp_path, dataset_id, data_type="csv",
        primary_file="counts.csv", species="Human",
    )
    step = FormatConverterStep(data_folder=str(tmp_path))
    result = step.convert(dataset_id)

    assert result is not None, f"convert() returned None: {result}"
    assert result["status"] == "success", f"Unexpected status: {result}"

    h5ad_path = os.path.join(str(tmp_path), "3.h5ad", f"{dataset_id}.h5ad")
    assert os.path.exists(h5ad_path), "h5ad file not written"

    adata = ad.read_h5ad(h5ad_path)
    assert adata.n_obs == 4
    assert adata.n_vars == 3
    assert "Dataset_id" in adata.obs.columns
    assert "Pubmed_id" in adata.obs.columns
    assert adata.obs["Dataset_id"].iloc[0] == dataset_id


def test_csv_gz_conversion_produces_h5ad(tmp_path):
    """A gzipped CSV should be converted correctly."""
    dataset_id = "99999999_04"
    _write_small_csv(tmp_path, dataset_id, "counts.csv.gz", compressed=True)
    _make_conversion_config(
        tmp_path, dataset_id, data_type="csv.gz",
        primary_file="counts.csv.gz", species="Mouse",
    )
    step = FormatConverterStep(data_folder=str(tmp_path))
    result = step.convert(dataset_id)

    assert result is not None
    assert result["status"] == "success"

    h5ad_path = os.path.join(str(tmp_path), "3.Mh5ad", f"{dataset_id}.h5ad")
    assert os.path.exists(h5ad_path)
    adata = ad.read_h5ad(h5ad_path)
    assert adata.n_obs == 4
    assert adata.n_vars == 3


def test_idempotent_skips_existing_h5ad(tmp_path):
    """Second call must return status='skipped' without re-converting."""
    dataset_id = "99999999_05"
    _write_small_csv(tmp_path, dataset_id, "counts.csv")
    _make_conversion_config(
        tmp_path, dataset_id, data_type="csv",
        primary_file="counts.csv", species="Human",
    )
    step = FormatConverterStep(data_folder=str(tmp_path))
    first = step.convert(dataset_id)
    assert first is not None and first["status"] == "success"

    second = step.convert(dataset_id)
    assert second is not None
    assert second["status"] in ("success", "skipped")


def test_looks_logged_detects_raw_counts(tmp_path):
    """looks_logged() should return False for integer count data."""
    X = np.array([[0, 100, 200], [50, 0, 999]], dtype=float)
    adata = ad.AnnData(X=X)
    assert SingleCellConverter.looks_logged(adata) is False


def test_looks_logged_detects_normalized():
    """looks_logged() should return True for log1p-transformed data."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, 8.0, (20, 50))
    adata = ad.AnnData(X=X)
    assert SingleCellConverter.looks_logged(adata) is True


def test_unknown_species_returns_failed(tmp_path):
    """convert() must return status='failed' for an unrecognised species."""
    dataset_id = "99999999_06"
    _write_small_csv(tmp_path, dataset_id, "counts.csv")
    _make_conversion_config(
        tmp_path, dataset_id, data_type="csv",
        primary_file="counts.csv", species="Zebrafish",
    )
    step = FormatConverterStep(data_folder=str(tmp_path))
    result = step.convert(dataset_id)
    assert result is not None
    assert result["status"] == "failed"


def test_10x_dir_resolved_from_bogus_primary_file(tmp_path):
    """Reproduces the reported bug: a 10x dataset whose primary_file is a
    description (because a *_RAW.tar was extracted) must still convert by
    resolving the extracted 10x directory on disk."""
    dataset_id = "99999999_07"
    # Simulate the downloader having extracted GSE_RAW.tar into a subdirectory.
    _write_10x_dir(tmp_path, dataset_id, "GSE147528_RAW")
    _make_conversion_config(
        tmp_path, dataset_id, data_type="10x",
        primary_file="GSE147528_RAW.tar -> extracted 10x matrix directory",
        species="Human",
        normalization_status="normalized",  # test fixture has low counts; skip empty-drop filter
    )
    step = FormatConverterStep(data_folder=str(tmp_path))
    result = step.convert(dataset_id)

    assert result is not None, "convert() returned None"
    assert result["status"] == "success", f"Unexpected status: {result}"

    h5ad_path = os.path.join(str(tmp_path), "3.h5ad", f"{dataset_id}.h5ad")
    assert os.path.exists(h5ad_path)
    adata = ad.read_h5ad(h5ad_path)
    assert adata.n_obs == 4
    assert adata.n_vars == 3


def test_multi_sample_merge_into_one_h5ad(tmp_path):
    """Several per-sample matrix files in a directory merge into one h5ad with
    an obs['sample'] column (GSE147528-style multi-sample dataset)."""
    import scipy.io as sio

    dataset_id = "99999999_08"
    raw = os.path.join(str(tmp_path), "2.raw", dataset_id, "GSE_RAW")
    os.makedirs(raw, exist_ok=True)
    # 4 cells x 3 genes, all cells non-empty.
    m = sp.csr_matrix(np.array([[1, 0, 3], [0, 5, 0], [2, 2, 2], [4, 1, 1]], dtype=float))
    sio.mmwrite(os.path.join(raw, "GSM1_SFG2.mtx"), m)
    sio.mmwrite(os.path.join(raw, "GSM2_EC1.mtx"), m)

    _make_conversion_config(
        tmp_path, dataset_id, data_type="mtx",
        primary_file="GSE_RAW.tar -> extracted matrices", species="Human",
    )
    step = FormatConverterStep(data_folder=str(tmp_path))
    result = step.convert(dataset_id)

    assert result is not None, "convert() returned None"
    assert result["status"] == "success", f"Unexpected status: {result}"
    assert result["n_samples"] == 2

    adata = ad.read_h5ad(os.path.join(str(tmp_path), "3.h5ad", f"{dataset_id}.h5ad"))
    assert adata.n_obs == 8  # 4 + 4
    assert "sample" in adata.obs.columns
    assert set(adata.obs["sample"].unique()) == {"GSM1_SFG2", "GSM2_EC1"}
