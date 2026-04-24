"""
System tests for DataDownloaderStep (Pipeline 2, Step A2).

Manifest fixtures are read from system_tests/data/1.manifest/.
Downloads are written to pytest tmp_path.

LLM is NOT required for this step — all tests run against live NCBI/GEO URLs.
The 'small file' tests use GSE233208_5XFAD_8samples.csv.gz (559 bytes) to keep
the test fast without mocking.
"""

import json
import logging
import os

import pytest

from src.data_processing.data_downloader import DataDownloaderStep, STATUS_FILENAME

logger = logging.getLogger(__name__)

MANIFEST_FIXTURE_DIR = "system_tests/data/1.manifest"

# Small GEO file used as a fast, real download target (559 bytes)
SMALL_GEO_FILE = "GSE233208_5XFAD_8samples.csv.gz"
SMALL_GEO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE233nnn/GSE233208/suppl/"
    "GSE233208_5XFAD_8samples.csv.gz"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_manifest(dataset_id: str, tmp_path, files: list[dict]) -> None:
    """Write a minimal manifest JSON into tmp_path/1.manifest/."""
    manifest_dir = os.path.join(str(tmp_path), "1.manifest")
    os.makedirs(manifest_dir, exist_ok=True)
    manifest = {
        "dataset_id": dataset_id,
        "pmid": "39578645",
        "accession_id": "GSE233208",
        "repository": "GEO",
        "listing_url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE233nnn/GSE233208/suppl/",
        "confirmed_format": "csv",
        "raw_data_available": False,
        "download_notes": None,
        "files": files,
    }
    with open(os.path.join(manifest_dir, f"{dataset_id}.json"), "w") as f:
        json.dump(manifest, f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_download_small_file(tmp_path):
    """Download a real 559-byte GEO CSV file and verify it lands on disk."""
    dataset_id = "test_small_download"
    _make_minimal_manifest(
        dataset_id, tmp_path,
        files=[{"filename": SMALL_GEO_FILE, "url": SMALL_GEO_URL,
                "purpose": "metadata", "format_hint": "csv", "notes": None}],
    )

    step = DataDownloaderStep(data_folder=str(tmp_path))
    status = step.download(dataset_id)

    assert status is not None
    assert status["all_success"] is True
    assert len(status["files"]) == 1

    record = status["files"][0]
    assert record["status"] == "success"
    assert record["size_bytes"] > 0

    local_path = os.path.join(str(tmp_path), "2.raw", dataset_id, SMALL_GEO_FILE)
    assert os.path.exists(local_path), "Downloaded file not found on disk"
    assert os.path.getsize(local_path) == record["size_bytes"]

    logger.info("Downloaded %s: %d bytes", SMALL_GEO_FILE, record["size_bytes"])


def test_download_writes_status_json(tmp_path):
    """download_status.json must be written alongside the downloaded file."""
    dataset_id = "test_status_json"
    _make_minimal_manifest(
        dataset_id, tmp_path,
        files=[{"filename": SMALL_GEO_FILE, "url": SMALL_GEO_URL,
                "purpose": "metadata", "format_hint": "csv", "notes": None}],
    )

    step = DataDownloaderStep(data_folder=str(tmp_path))
    step.download(dataset_id)

    status_path = os.path.join(str(tmp_path), "2.raw", dataset_id, STATUS_FILENAME)
    assert os.path.exists(status_path)
    with open(status_path) as f:
        status = json.load(f)
    assert status["dataset_id"] == dataset_id
    assert "downloaded_at" in status
    assert isinstance(status["files"], list)


def test_download_skips_complete_file(tmp_path):
    """If the file already exists with matching size, status must be 'skipped'."""
    dataset_id = "test_skip"
    _make_minimal_manifest(
        dataset_id, tmp_path,
        files=[{"filename": SMALL_GEO_FILE, "url": SMALL_GEO_URL,
                "purpose": "metadata", "format_hint": "csv", "notes": None}],
    )

    step = DataDownloaderStep(data_folder=str(tmp_path))
    # First download
    first = step.download(dataset_id)
    assert first["all_success"] is True

    # Delete the status file to force re-evaluation (simulates restart)
    status_path = os.path.join(str(tmp_path), "2.raw", dataset_id, STATUS_FILENAME)
    os.remove(status_path)

    # Second download — file already present with correct size → should skip
    second = step.download(dataset_id)
    assert second is not None
    assert second["files"][0]["status"] == "skipped"
    logger.info("Skip behaviour confirmed for %s", SMALL_GEO_FILE)


def test_download_no_manifest_returns_none(tmp_path):
    """Missing manifest must return None without raising."""
    step = DataDownloaderStep(data_folder=str(tmp_path))
    result = step.download("nonexistent_dataset_id")
    assert result is None


def test_download_bad_url_marks_failed(tmp_path):
    """An unreachable URL must result in status='failed', not an exception."""
    dataset_id = "test_bad_url"
    _make_minimal_manifest(
        dataset_id, tmp_path,
        files=[{"filename": "nonexistent.csv.gz",
                "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE233nnn/GSE233208/suppl/nonexistent.csv.gz",
                "purpose": "metadata", "format_hint": "csv", "notes": None}],
    )

    step = DataDownloaderStep(data_folder=str(tmp_path))
    status = step.download(dataset_id)

    assert status is not None
    assert status["all_success"] is False
    assert status["files"][0]["status"] == "failed"
    assert status["files"][0]["error"] is not None
    logger.info("Bad-URL error: %s", status["files"][0]["error"])


@pytest.mark.skip()
def test_download_fixture_manifest_dataset_03(tmp_path):
    """
    End-to-end test using the real fixture manifest for dataset 39578645_03.
    Downloads the small metadata CSV only (skips the 3.9 GB RDS).
    Requires internet access; skipped by default to avoid large downloads.
    """
    dataset_id = "39578645_03"
    # Copy fixture manifest into tmp_path
    src = os.path.join(MANIFEST_FIXTURE_DIR, f"{dataset_id}.json")
    with open(src) as f:
        manifest = json.load(f)

    # Override: keep only the small CSV file for the test
    manifest["files"] = [
        f for f in manifest["files"]
        if f["format_hint"] == "csv"
    ]
    manifest_dir = os.path.join(str(tmp_path), "1.manifest")
    os.makedirs(manifest_dir, exist_ok=True)
    with open(os.path.join(manifest_dir, f"{dataset_id}.json"), "w") as f:
        json.dump(manifest, f)

    step = DataDownloaderStep(data_folder=str(tmp_path))
    status = step.download(dataset_id)

    assert status is not None
    assert status["all_success"] is True
    logger.info("Fixture manifest download status: %s", status)
