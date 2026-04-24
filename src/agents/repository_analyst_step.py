"""
Pipeline 2, Step A1 — RepositoryAnalystStep

Input  : one dataset dict from data/0.metadata/{pmid}.json
Output : data/1.manifest/{dataset_id}.json

The LLM is given:
  - dataset metadata (species, technology, format hint, accession IDs)
  - the actual file listing fetched from GEO/SRA/Zenodo

It decides which files are the raw count matrices (vs. processed/metadata files),
confirms the data format, and writes a download manifest.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field

from .common_agent import CommonAgent
from ..paper_query.repository_fetcher import (
    RepoListing,
    fetch_geo_suppl_listing,
    search_geo_by_pmid,
)

logger = logging.getLogger(__name__)

REPOSITORY_ANALYST_OUTPUT_SUBDIR = "1.manifest"

REPOSITORY_ANALYST_SYSTEM_PROMPT = ChatPromptTemplate.from_template("""
You are an expert bioinformatician helping to build an automated single-cell data processing pipeline.

You will be given:
1. Metadata about a dataset from a published paper (species, technology, expected format, etc.)
2. The actual file listing fetched from the public repository where the data was deposited.

Your task is to analyse the file listing and decide:
- Which files should be downloaded to obtain the **raw count matrix** for this dataset.
- What the **actual data format** is (may differ from the expected hint if only processed files exist).
- Whether **raw counts are truly available** or only processed/normalized data was deposited.

Guidelines:
- **Prefer raw count matrices** over processed objects (e.g., prefer Cell Ranger output over Seurat RDS).
- Files named `*_RAW.tar`, `*matrix.mtx.gz`, `*barcodes.tsv.gz`, `*features.tsv.gz`, `*.h5` from
  Cell Ranger are raw count data.
- Files named `*seurat*.rds`, `*processed*.rds`, `*integrated*.rds`, `*annotated*.rds` are processed.
- Files named `*_meta*.csv`, `*_sample*.csv` are metadata/annotation, not count matrices.
- If **only processed files are available**, set `raw_data_available` to false and still list the
  most useful processed file for download (e.g., the Seurat RDS with raw counts stored inside).
- Use the species and technology hints to select files relevant to **this specific dataset** when
  the accession contains data for multiple datasets (e.g., both human and mouse).

---

**Dataset metadata:**
- dataset_id   : {dataset_id}
- species      : {species}
- technology   : {technology}
- format hint  : {data_format_hint}
- accession IDs: {accession_ids}
- repository   : {repository}
- norm. hint   : {normalization_hint}
- tissue       : {tissue_type}
- notes        : {notes}

**Repository file listing ({listing_url}):**
{file_listing}
""")


class ManifestFile(BaseModel):
    filename: str = Field(description="Exact filename as it appears in the repository listing")
    url: str = Field(description="Full download URL")
    purpose: str = Field(
        description="Role of this file: 'count_matrix', 'barcodes', 'features', "
                    "'metadata', 'processed', or 'other'"
    )
    format_hint: str = Field(
        description="File format: '10x_mtx', '10x_h5', 'csv', 'tsv', 'h5ad', "
                    "'rds', 'archive', 'other'"
    )
    notes: Optional[str] = Field(default=None, description="Any special handling notes")


class RepositoryAnalysisResult(BaseModel):
    reasoning_process: str = Field(
        description="Step-by-step reasoning for selecting files and determining format"
    )
    files: List[ManifestFile] = Field(
        description="Files to download for this dataset (raw count matrices first)"
    )
    confirmed_format: str = Field(
        description="Definitive data format after inspecting the listing: "
                    "'10x_mtx', '10x_h5', 'csv', 'tsv', 'h5ad', 'rds', 'archive', 'unknown'"
    )
    raw_data_available: bool = Field(
        description="True if raw (unnormalized) count matrices are available for download"
    )
    download_notes: Optional[str] = Field(
        default=None,
        description="Special instructions for the download step (e.g., 'extract RAW.tar first')"
    )


class RepositoryAnalystStep:
    """
    Pipeline 2, Step A1: fetch repository file listings and produce a download manifest.

    Input  : dataset dict (one entry from 0.metadata/{pmid}.json)
    Output : data/1.manifest/{dataset_id}.json
    """

    def __init__(
        self,
        llm: BaseChatOpenAI,
        data_folder: Optional[str] = None,
        ncbi_email: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.data_folder = data_folder or os.getenv("DATA_FOLDER", ".")
        self.ncbi_email = ncbi_email or os.getenv("NCBI_EMAIL")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, dataset: dict) -> Optional[dict]:
        """
        Analyze one dataset entry and produce its download manifest.

        Returns the manifest dict, or None on failure.
        Skips the LLM call if the manifest file already exists.
        """
        dataset_id = dataset["dataset_id"]
        out_path = self._output_path(dataset_id)

        if os.path.exists(out_path):
            logger.info("RepositoryAnalystStep: skipping %s (manifest exists)", dataset_id)
            with open(out_path) as f:
                return json.load(f)

        listing = self._fetch_listing(dataset)
        if listing is None:
            logger.warning(
                "RepositoryAnalystStep: no repository listing for %s — writing empty manifest",
                dataset_id,
            )
            return self._write_empty_manifest(dataset, out_path)

        result = self._run_llm(dataset, listing)
        if result is None:
            logger.error("RepositoryAnalystStep: LLM returned None for %s", dataset_id)
            return None

        manifest = self._build_manifest(dataset, listing, result)
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(
            "RepositoryAnalystStep: wrote %s (%d file(s), format=%s, raw=%s)",
            out_path, len(manifest["files"]),
            manifest["confirmed_format"], manifest["raw_data_available"],
        )
        return manifest

    # ------------------------------------------------------------------
    # Repository fetching
    # ------------------------------------------------------------------

    def _fetch_listing(self, dataset: dict) -> Optional[RepoListing]:
        """
        Return a RepoListing for the dataset's accession.
        Falls back to a GEO search by PMID if accession_ids is empty.
        """
        accession_ids: list = dataset.get("accession_ids") or []
        repository: str = (dataset.get("repository") or "").upper()
        pmid: str = dataset.get("pmid", "")

        # --- Try accession IDs from metadata first ---
        for acc in accession_ids:
            if acc.upper().startswith("GSE") or repository == "GEO":
                listing = fetch_geo_suppl_listing(acc, )
                if not listing.error and listing.files:
                    return listing
            # Additional repository types (SRA, Zenodo) can be added here.

        # --- Fallback: search GEO by PMID ---
        if pmid:
            found = search_geo_by_pmid(pmid, email=self.ncbi_email)
            for acc in found:
                listing = fetch_geo_suppl_listing(acc)
                if not listing.error and listing.files:
                    # Patch accession into dataset so the manifest is complete
                    dataset.setdefault("accession_ids", [])
                    if acc not in dataset["accession_ids"]:
                        dataset["accession_ids"].append(acc)
                    return listing

        return None

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _run_llm(
        self, dataset: dict, listing: RepoListing
    ) -> Optional[RepositoryAnalysisResult]:
        file_lines = "\n".join(
            f"  {f.filename}  ({f.size or '?'})" for f in listing.files
        ) or "  (no files found)"

        system_prompt = REPOSITORY_ANALYST_SYSTEM_PROMPT.format(
            dataset_id=dataset.get("dataset_id", "N/A"),
            species=dataset.get("species", "N/A"),
            technology=dataset.get("technology", "N/A"),
            data_format_hint=dataset.get("data_format_hint", "unknown"),
            accession_ids=", ".join(dataset.get("accession_ids") or []) or "none",
            repository=dataset.get("repository", "unknown"),
            normalization_hint=dataset.get("normalization_hint", "unknown"),
            tissue_type=dataset.get("tissue_type", "N/A"),
            notes=dataset.get("notes") or "none",
            listing_url=listing.listing_url,
            file_listing=file_lines,
        )
        agent = CommonAgent(llm=self.llm)
        res, _, _, _ = agent.go(
            system_prompt=system_prompt,
            instruction_prompt=(
                "Analyse the file listing. Select the files needed for the raw count matrix "
                "of this specific dataset and fill in the structured output."
            ),
            schema=RepositoryAnalysisResult,
        )
        return res

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _output_path(self, dataset_id: str) -> str:
        out_dir = os.path.join(self.data_folder, REPOSITORY_ANALYST_OUTPUT_SUBDIR)
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, f"{dataset_id}.json")

    @staticmethod
    def _build_manifest(
        dataset: dict,
        listing: RepoListing,
        result: RepositoryAnalysisResult,
    ) -> dict:
        return {
            "dataset_id": dataset["dataset_id"],
            "pmid": dataset.get("pmid", ""),
            "accession_id": listing.accession_id,
            "repository": dataset.get("repository", "unknown"),
            "listing_url": listing.listing_url,
            "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reasoning": result.reasoning_process,
            "files": [f.model_dump() for f in result.files],
            "confirmed_format": result.confirmed_format,
            "raw_data_available": result.raw_data_available,
            "download_notes": result.download_notes,
        }

    @staticmethod
    def _write_empty_manifest(dataset: dict, out_path: str) -> dict:
        manifest = {
            "dataset_id": dataset["dataset_id"],
            "pmid": dataset.get("pmid", ""),
            "accession_id": None,
            "repository": dataset.get("repository", "unknown"),
            "listing_url": None,
            "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reasoning": "No repository listing could be fetched.",
            "files": [],
            "confirmed_format": "unknown",
            "raw_data_available": False,
            "download_notes": "No accession ID found and GEO search by PMID returned no results.",
        }
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest
