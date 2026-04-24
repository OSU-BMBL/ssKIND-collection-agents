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

logger = logging.getLogger(__name__)

METADATA_EXTRACTION_OUTPUT_SUBDIR = "0.metadata"

METADATA_EXTRACTION_SYSTEM_PROMPT = ChatPromptTemplate.from_template("""
You are an expert biomedical data scientist specializing in single-cell RNA sequencing (scRNA-seq)
and spatial transcriptomics.

You will be given the title and full text of a scientific paper that has been confirmed to contain
**original, publicly accessible** single-cell or spatial transcriptomics data.

Your task is to identify and extract structured metadata for **each distinct dataset** deposited by
this paper. A paper may contribute multiple datasets (e.g., different cohorts, species, or
sequencing technologies).

For each dataset extract the following fields:

- **species**: "Human", "Mouse", or "Other"
- **technology**: The sequencing or profiling technology (e.g., "10x Chromium", "SMART-seq2",
  "Visium", "Slide-seq", "MERFISH", "GeoMx")
- **data_format_hint**: The expected raw file format — one of: "10x_mtx" (MTX + barcodes +
  features directory), "10x_h5" (.h5 from Cell Ranger), "csv", "tsv", "h5ad", "rds" (Seurat),
  "other", or "unknown"
- **accession_ids**: List of repository accession IDs exactly as written in the paper (e.g.,
  "GSE123456", "SRP789012", "10.5281/zenodo.1234567")
- **repository**: Primary repository where raw data is deposited — one of: "GEO", "SRA",
  "Zenodo", "ArrayExpress", "Dryad", "figshare", "other"
- **normalization_hint**: Whether the deposited data is raw counts, normalized, or unknown —
  one of: "raw_counts", "normalized", "unknown"
- **n_samples**: Number of biological samples or donors as an integer, or null if not reported
- **n_cells_reported**: Total number of cells or spots reported in the paper as an integer,
  or null if not reported
- **atlas_eligible**: true only if (1) species is Human or Mouse, (2) tissue is brain-related,
  and (3) n_cells_reported is null or > 200
- **tissue_type**: Specific tissue or brain region (e.g., "prefrontal cortex", "hippocampus",
  "entorhinal cortex", "whole brain", "spinal cord")
- **notes**: Any important caveats, e.g., data is split across multiple accessions, only
  processed data is deposited, mixed species, restricted-access human data with public mouse
  counterpart, etc.

---

**Title:**
{title}

**Full Text:**
{full_text}
""")


class ExtractedDataset(BaseModel):
    species: str = Field(
        description="Species of the dataset: 'Human', 'Mouse', or 'Other'"
    )
    technology: str = Field(
        description="Sequencing/profiling technology (e.g., '10x Chromium', 'Visium', 'SMART-seq2')"
    )
    data_format_hint: str = Field(
        description="Expected raw file format: '10x_mtx', '10x_h5', 'csv', 'tsv', 'h5ad', 'rds', 'other', or 'unknown'"
    )
    accession_ids: List[str] = Field(
        description="Repository accession IDs exactly as written in the paper"
    )
    repository: str = Field(
        description="Primary repository: 'GEO', 'SRA', 'Zenodo', 'ArrayExpress', 'Dryad', 'figshare', or 'other'"
    )
    normalization_hint: str = Field(
        description="Normalization status of deposited data: 'raw_counts', 'normalized', or 'unknown'"
    )
    n_samples: Optional[int] = Field(
        default=None,
        description="Number of biological samples/donors, or null if not reported"
    )
    n_cells_reported: Optional[int] = Field(
        default=None,
        description="Total cells or spots reported in the paper, or null if not reported"
    )
    atlas_eligible: bool = Field(
        description="True if Human or Mouse brain data with n_cells > 200, suitable for atlas integration"
    )
    tissue_type: str = Field(
        description="Tissue or brain region (e.g., 'prefrontal cortex', 'hippocampus', 'whole brain')"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional caveats or relevant information about this dataset"
    )


class MetadataExtractionResult(BaseModel):
    reasoning_process: str = Field(
        description="Step-by-step reasoning used to identify and characterize each dataset"
    )
    datasets: List[ExtractedDataset] = Field(
        description="List of distinct datasets found in the paper"
    )


class MetadataExtractorStep:
    """
    Pipeline 2, Step 0: extract structured dataset metadata from accepted paper full text.

    Input  : pmid, title, full_text (from Pipeline 1 DB)
    Output : data/0.metadata/{pmid}.json   (intermediate file)

    Skips extraction if the output file already exists (safe to re-run).
    """

    def __init__(
        self,
        llm: BaseChatOpenAI,
        data_folder: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.data_folder = data_folder or os.getenv("DATA_FOLDER", ".")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        pmid: str,
        title: str,
        full_text: str,
    ) -> Optional[List[dict]]:
        """
        Extract dataset metadata from paper text.

        Returns the list of dataset dicts written to the JSON file,
        or None if the LLM call failed.
        Skips the LLM call and reads from disk if the output already exists.
        """
        out_path = self._output_path(pmid)

        if os.path.exists(out_path):
            logger.info("MetadataExtractorStep: %s already extracted, loading from disk", pmid)
            with open(out_path) as f:
                return json.load(f)["datasets"]

        result = self._run_llm(title, full_text)
        if result is None:
            logger.error("MetadataExtractorStep: LLM returned None for PMID %s", pmid)
            return None

        datasets = self._assign_dataset_ids(pmid, result.datasets)
        self._write(pmid, result.reasoning_process, datasets, out_path)
        logger.info(
            "MetadataExtractorStep: wrote %s (%d dataset(s))", out_path, len(datasets)
        )
        return datasets

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _output_path(self, pmid: str) -> str:
        out_dir = os.path.join(self.data_folder, METADATA_EXTRACTION_OUTPUT_SUBDIR)
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, f"{pmid}.json")

    def _run_llm(
        self, title: str, full_text: str
    ) -> Optional[MetadataExtractionResult]:
        system_prompt = METADATA_EXTRACTION_SYSTEM_PROMPT.format(
            title=title,
            full_text=full_text,
        )
        agent = CommonAgent(llm=self.llm)
        res, _, _, _ = agent.go(
            system_prompt=system_prompt,
            instruction_prompt=(
                "Identify every distinct dataset in this paper. "
                "Reason step by step, then fill in the structured output."
            ),
            schema=MetadataExtractionResult,
        )
        return res

    @staticmethod
    def _assign_dataset_ids(pmid: str, datasets: List[ExtractedDataset]) -> List[dict]:
        result = []
        for i, ds in enumerate(datasets, start=1):
            d = ds.model_dump()
            d["dataset_id"] = f"{pmid}_{i:02d}"
            result.append(d)
        return result

    @staticmethod
    def _write(pmid: str, reasoning: str, datasets: List[dict], out_path: str) -> None:
        record = {
            "pmid": pmid,
            "extracted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reasoning": reasoning,
            "datasets": datasets,
        }
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
