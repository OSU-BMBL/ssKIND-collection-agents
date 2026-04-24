"""
System tests for MetadataExtractorStep (Pipeline 2, Step 0).

These tests consume real API tokens and write to DATA_FOLDER/0.metadata/.
Skip markers are in place — remove @pytest.mark.skip() to run individually.
"""

import json
import logging
import os
import pytest

from src.agents.metadata_extractor_step import MetadataExtractorStep
from src.paper_query.pubmed_query import query_title_abstract_ispreprint
from src.workflow.workflow_utils import obtain_full_text

logger = logging.getLogger(__name__)


def _run_extraction(llm, pmid: str, tmp_path) -> list[dict]:
    """Helper: fetch paper text, run extraction, return datasets list."""
    title, abstract, _ = query_title_abstract_ispreprint(pmid)
    full_text = obtain_full_text(pmid)
    assert title is not None and full_text, f"Could not retrieve text for PMID {pmid}"

    step = MetadataExtractorStep(llm=llm, data_folder=str(tmp_path))
    datasets = step.extract(pmid=pmid, title=title, full_text=full_text)
    return datasets


# @pytest.mark.skip()
@pytest.mark.parametrize("pmid", [
    # "39578645",  # known accepted: original + relevant AD scRNA paper
    "39607927",  # known accepted: original + relevant AD spatial paper
])
def test_extraction_produces_valid_json(llm, tmp_path, pmid):
    """Output JSON must exist and contain at least one dataset with required fields."""
    datasets = _run_extraction(llm, pmid, tmp_path)

    assert datasets is not None, "extract() returned None"
    assert len(datasets) >= 1, "Expected at least one dataset"

    required_fields = {
        "dataset_id", "species", "technology", "data_format_hint",
        "accession_ids", "repository", "normalization_hint",
        "atlas_eligible", "tissue_type",
    }
    for ds in datasets:
        missing = required_fields - ds.keys()
        assert not missing, f"Dataset missing fields: {missing}"
        assert isinstance(ds["accession_ids"], list), "accession_ids must be a list"
        assert ds["species"] in ("Human", "Mouse", "Other"), \
            f"Unexpected species: {ds['species']}"
        assert ds["normalization_hint"] in ("raw_counts", "normalized", "unknown"), \
            f"Unexpected normalization_hint: {ds['normalization_hint']}"

    # Verify the JSON file was written to disk
    out_file = os.path.join(str(tmp_path), "0.metadata", f"{pmid}.json")
    assert os.path.exists(out_file), f"Output file not found: {out_file}"
    with open(out_file) as f:
        record = json.load(f)
    assert record["pmid"] == pmid
    assert len(record["datasets"]) == len(datasets)

    logger.info("PMID %s — %d dataset(s) extracted:", pmid, len(datasets))
    for ds in datasets:
        logger.info(
            "  [%s] species=%s technology=%s accessions=%s atlas_eligible=%s",
            ds["dataset_id"], ds["species"], ds["technology"],
            ds["accession_ids"], ds["atlas_eligible"],
        )


@pytest.mark.skip()
def test_extraction_is_idempotent(llm, tmp_path):
    """Running extraction twice on the same PMID must skip the LLM and return same data."""
    pmid = "39578645" # "39329069"
    title, _, _ = query_title_abstract_ispreprint(pmid)
    full_text = obtain_full_text(pmid)

    step = MetadataExtractorStep(llm=llm, data_folder=str(tmp_path))

    first = step.extract(pmid=pmid, title=title, full_text=full_text)
    assert first is not None

    # Second call must read from disk (no LLM call, same result)
    second = step.extract(pmid=pmid, title=title, full_text=full_text)
    assert second == first, "Idempotency check failed — results differ on second run"
    logger.info("Idempotency check passed for PMID %s", pmid)
