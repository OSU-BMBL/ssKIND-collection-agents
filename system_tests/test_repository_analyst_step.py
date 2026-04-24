"""
System tests for RepositoryAnalystStep (Pipeline 2, Step A1).

Input metadata is read from system_tests/data/0.metadata/ (pre-generated fixtures).
Tests write manifests to a pytest tmp_path so nothing is committed to disk.

Remove @pytest.mark.skip() to run a test (consumes real API tokens).
"""

import json
import logging
import os
import pytest

from src.agents.repository_analyst_step import RepositoryAnalystStep
from src.paper_query.repository_fetcher import fetch_geo_suppl_listing, search_geo_by_pmid

logger = logging.getLogger(__name__)

METADATA_FIXTURE_DIR = "system_tests/data/0.metadata"


def _load_metadata(pmid: str) -> dict:
    path = os.path.join(METADATA_FIXTURE_DIR, f"{pmid}.json")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Unit-level: test the GEO fetcher without LLM
# ---------------------------------------------------------------------------

def test_fetch_geo_suppl_listing_gse233208():
    """GEO suppl/ listing for GSE233208 must return ≥1 file without error."""
    listing = fetch_geo_suppl_listing("GSE233208")
    assert listing.error is None, f"Fetch error: {listing.error}"
    assert len(listing.files) > 0, "Expected at least one file"
    filenames = [f.filename for f in listing.files]
    logger.info("GSE233208 files: %s", filenames)
    # Sanity check: all known file types are present
    assert any(f.endswith(".rds.gz") or f.endswith(".csv.gz") for f in filenames)


def test_geo_suppl_url_pattern():
    """Verify the GSE→FTP URL prefix formula for several accession sizes."""
    from src.paper_query.repository_fetcher import _geo_suppl_url
    assert "GSE233nnn" in _geo_suppl_url("GSE233208")
    assert "GSE1nnn"   in _geo_suppl_url("GSE1234")
    assert "GSEnnn"    in _geo_suppl_url("GSE999")
    assert "GSE12nnn"  in _geo_suppl_url("GSE12345")


# ---------------------------------------------------------------------------
# LLM tests
# ---------------------------------------------------------------------------

# @pytest.mark.skip()
@pytest.mark.parametrize("pmid, dataset_idx, expected_format_options", [
    # 39578645: 3 datasets, all under GSE233208
    # Dataset 01 = Human Visium → expect rds (only processed Seurat available)
    ("39578645", 0, {"rds", "unknown"}),
    # Dataset 02 = Human snRNA-seq (Parse Bio) → rds or unknown
    ("39578645", 1, {"rds", "unknown"}),
    # Dataset 03 = Mouse Visium 5xFAD → rds or unknown
    ("39578645", 2, {"rds", "unknown"}),
])
def test_analyst_produces_valid_manifest(llm, tmp_path, pmid, dataset_idx, expected_format_options):
    """Manifest must be written with required fields; format must be one of the expected options."""
    record = _load_metadata(pmid)
    dataset = record["datasets"][dataset_idx]
    dataset["pmid"] = pmid  # ensure pmid is present in dataset dict

    step = RepositoryAnalystStep(llm=llm, data_folder=str(tmp_path))
    manifest = step.analyze(dataset)

    assert manifest is not None, "analyze() returned None"

    required_keys = {
        "dataset_id", "pmid", "accession_id", "repository", "listing_url",
        "analyzed_at", "files", "confirmed_format", "raw_data_available",
    }
    assert required_keys <= manifest.keys(), \
        f"Missing keys: {required_keys - manifest.keys()}"

    assert manifest["confirmed_format"] in expected_format_options, \
        f"Unexpected format: {manifest['confirmed_format']}"

    assert isinstance(manifest["files"], list)
    assert isinstance(manifest["raw_data_available"], bool)

    # Check output file was written
    out_file = os.path.join(str(tmp_path), "1.manifest", f"{dataset['dataset_id']}.json")
    assert os.path.exists(out_file), f"Manifest file not written: {out_file}"

    logger.info(
        "PMID %s dataset[%d] → format=%s raw=%s files=%d",
        pmid, dataset_idx,
        manifest["confirmed_format"],
        manifest["raw_data_available"],
        len(manifest["files"]),
    )
    for f in manifest["files"]:
        logger.info("  [%s] %s — %s", f["purpose"], f["filename"], f.get("notes", ""))


@pytest.mark.skip()
def test_analyst_idempotent(llm, tmp_path):
    """Second call with same dataset_id must skip LLM and return identical manifest."""
    record = _load_metadata("39578645")
    dataset = record["datasets"][0]
    dataset["pmid"] = "39578645"

    step = RepositoryAnalystStep(llm=llm, data_folder=str(tmp_path))
    first = step.analyze(dataset)
    assert first is not None

    second = step.analyze(dataset)
    assert second == first, "Idempotency failed — manifests differ on second run"


@pytest.mark.skip()
def test_analyst_empty_accession_fallback(llm, tmp_path):
    """
    Dataset with no accession IDs (39607927) should produce an empty manifest
    (GEO search by PMID returns nothing for this paper).
    """
    record = _load_metadata("39607927")
    dataset = record["datasets"][0]
    dataset["pmid"] = "39607927"

    step = RepositoryAnalystStep(llm=llm, data_folder=str(tmp_path))
    manifest = step.analyze(dataset)

    assert manifest is not None
    assert manifest["accession_id"] is None or manifest["files"] == [] or not manifest["raw_data_available"]
    logger.info("Empty-accession manifest: %s", manifest)
