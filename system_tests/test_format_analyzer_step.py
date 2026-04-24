"""
System tests for FormatAnalyzerStep (Pipeline 2, Step B1).

Fixture layout expected under system_tests/data/:
  1.manifest/39578645_01.json   (manifest fixture)
  2.raw/39578645_01/download_status.json  (download status fixture)
  2.raw/39578645_03/download_status.json  (has small CSV for text-peek test)

LLM tests are skipped by default — remove @pytest.mark.skip() to run.
"""

import json
import logging
import os
import shutil

import pytest

from src.agents.format_analyzer_step import (
    FormatAnalyzerStep,
    CONVERSION_CONFIG_FILENAME,
    VALID_DATA_TYPES,
)

logger = logging.getLogger(__name__)

FIXTURE_ROOT = "system_tests/data"
MANIFEST_FIXTURE_DIR = os.path.join(FIXTURE_ROOT, "1.manifest")
RAW_FIXTURE_DIR = os.path.join(FIXTURE_ROOT, "2.raw")


def _copy_fixtures(dataset_id: str, tmp_path) -> None:
    """Copy manifest + download_status fixtures into tmp_path directory tree."""
    # manifest
    manifest_dst = os.path.join(str(tmp_path), "1.manifest")
    os.makedirs(manifest_dst, exist_ok=True)
    shutil.copy(
        os.path.join(MANIFEST_FIXTURE_DIR, f"{dataset_id}.json"),
        os.path.join(manifest_dst, f"{dataset_id}.json"),
    )
    # download_status
    raw_dst = os.path.join(str(tmp_path), "2.raw", dataset_id)
    os.makedirs(raw_dst, exist_ok=True)
    shutil.copy(
        os.path.join(RAW_FIXTURE_DIR, dataset_id, "download_status.json"),
        os.path.join(raw_dst, "download_status.json"),
    )


# ---------------------------------------------------------------------------
# Non-LLM unit tests
# ---------------------------------------------------------------------------

def test_missing_manifest_returns_none(tmp_path):
    """analyze() must return None (not raise) when the manifest is missing."""
    # Only copy download_status, no manifest
    raw_dst = os.path.join(str(tmp_path), "2.raw", "39578645_01")
    os.makedirs(raw_dst, exist_ok=True)
    shutil.copy(
        os.path.join(RAW_FIXTURE_DIR, "39578645_01", "download_status.json"),
        os.path.join(raw_dst, "download_status.json"),
    )
    from unittest.mock import MagicMock
    step = FormatAnalyzerStep(llm=MagicMock(), data_folder=str(tmp_path))
    assert step.analyze("39578645_01") is None


def test_missing_download_status_returns_none(tmp_path):
    """analyze() must return None (not raise) when download_status is missing."""
    manifest_dst = os.path.join(str(tmp_path), "1.manifest")
    os.makedirs(manifest_dst, exist_ok=True)
    shutil.copy(
        os.path.join(MANIFEST_FIXTURE_DIR, "39578645_01.json"),
        os.path.join(manifest_dst, "39578645_01.json"),
    )
    from unittest.mock import MagicMock
    step = FormatAnalyzerStep(llm=MagicMock(), data_folder=str(tmp_path))
    assert step.analyze("39578645_01") is None


def test_text_peek_reads_small_csv(tmp_path):
    """
    When a small CSV (< 10 KB) is present on disk the step should include its
    first lines in the LLM prompt. We verify the internal helper directly.
    """
    # Download the real small GEO CSV so it's present on disk
    import requests
    url = (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE233nnn/GSE233208/suppl/"
        "GSE233208_5XFAD_8samples.csv.gz"
    )
    raw_dir = os.path.join(str(tmp_path), "2.raw", "39578645_03")
    os.makedirs(raw_dir, exist_ok=True)
    local_csv = os.path.join(raw_dir, "GSE233208_5XFAD_8samples.csv.gz")
    with requests.get(url, timeout=15, stream=True) as r:
        r.raise_for_status()
        with open(local_csv, "wb") as fh:
            for chunk in r.iter_content(1024):
                fh.write(chunk)

    # Build a minimal download_status pointing to this file
    status = {
        "dataset_id": "39578645_03",
        "files": [
            {
                "filename": "GSE233208_5XFAD_8samples.csv.gz",
                "status": "success",
                "size_bytes": os.path.getsize(local_csv),
            }
        ],
        "all_success": True,
    }
    with open(os.path.join(raw_dir, "download_status.json"), "w") as f:
        json.dump(status, f)

    from unittest.mock import MagicMock
    step = FormatAnalyzerStep(llm=MagicMock(), data_folder=str(tmp_path))
    preview = step._build_text_previews("39578645_03", status)

    assert "GSE233208_5XFAD_8samples.csv.gz" in preview, \
        "Expected filename in preview"
    assert len(preview) > 10, "Preview should contain actual CSV content"
    logger.info("Text preview:\n%s", preview)


# ---------------------------------------------------------------------------
# LLM tests
# ---------------------------------------------------------------------------

@pytest.mark.skip()
@pytest.mark.parametrize("dataset_id, expected_data_types, expected_r_extraction", [
    # Human Visium — Seurat RDS → rds, requires R
    ("39578645_01", {"rds"}, True),
    # Human snRNA-seq — Seurat RDS → rds, requires R
    ("39578645_02", {"rds"}, True),
    # Mouse Visium — Seurat RDS → rds, requires R
    ("39578645_03", {"rds"}, True),
])
def test_analyzer_produces_valid_config(
    llm, tmp_path, dataset_id, expected_data_types, expected_r_extraction
):
    """Config must have required fields; data_type and r_extraction must match expectations."""
    _copy_fixtures(dataset_id, tmp_path)

    step = FormatAnalyzerStep(llm=llm, data_folder=str(tmp_path))
    config = step.analyze(dataset_id)

    assert config is not None, "analyze() returned None"

    required_keys = {
        "dataset_id", "pmid", "analyzed_at", "data_type",
        "primary_file", "species", "gene_mapping_needed",
        "normalization_status", "requires_r_extraction",
    }
    assert required_keys <= config.keys(), \
        f"Missing keys: {required_keys - config.keys()}"

    assert config["data_type"] in VALID_DATA_TYPES, \
        f"Invalid data_type: {config['data_type']}"
    assert config["data_type"] in expected_data_types, \
        f"Expected one of {expected_data_types}, got {config['data_type']}"
    assert config["requires_r_extraction"] == expected_r_extraction, \
        f"Expected requires_r_extraction={expected_r_extraction}"
    assert config["normalization_status"] in ("raw_counts", "normalized", "unknown")
    assert config["species"] in ("Human", "Mouse", "Other")

    # Config file written to disk
    config_path = os.path.join(
        str(tmp_path), "2.raw", dataset_id, CONVERSION_CONFIG_FILENAME
    )
    assert os.path.exists(config_path)

    logger.info(
        "%s → data_type=%s species=%s norm=%s r_extract=%s",
        dataset_id, config["data_type"], config["species"],
        config["normalization_status"], config["requires_r_extraction"],
    )


@pytest.mark.skip()
def test_analyzer_idempotent(llm, tmp_path):
    """Second call must skip LLM and return the same config."""
    dataset_id = "39578645_01"
    _copy_fixtures(dataset_id, tmp_path)

    step = FormatAnalyzerStep(llm=llm, data_folder=str(tmp_path))
    first = step.analyze(dataset_id)
    assert first is not None

    second = step.analyze(dataset_id)
    assert second == first, "Idempotency failed — configs differ on second run"
