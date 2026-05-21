"""
Pipeline 2, Step B1 — FormatAnalyzerStep

Input  : data/1.manifest/{dataset_id}.json
         data/2.raw/{dataset_id}/download_status.json
         data/2.raw/{dataset_id}/ (file listing; small text files read for context)
Output : data/2.raw/{dataset_id}/conversion_config.json

The LLM receives:
  - Dataset metadata from the manifest (species, technology, confirmed_format)
  - The list of downloaded files with sizes (from download_status.json)
  - Content snippet of any small text file (< TEXT_SNIFF_LIMIT bytes) that was downloaded

It decides the exact conversion strategy to pass to FormatConverterStep (Step B2).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field

from .common_agent import CommonAgent
from ..data_processing.archive_utils import resolve_input_path

logger = logging.getLogger(__name__)

MANIFEST_SUBDIR = "1.manifest"
RAW_SUBDIR = "2.raw"
DOWNLOAD_STATUS_FILENAME = "download_status.json"
CONVERSION_CONFIG_FILENAME = "conversion_config.json"

# Files smaller than this threshold have their content peeked for context
TEXT_SNIFF_LIMIT = 10_000  # bytes
TEXT_EXTENSIONS = {".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz"}

# Valid data_type values accepted by SingleCellConverter.read_data()
VALID_DATA_TYPES = {
    "10x",          # Cell Ranger MTX directory
    "10x_matrix",   # single MTX file
    "csv",
    "tsv",
    "txt",
    "csv.gz",
    "tsv.gz",
    "mtx",
    "h5",           # Cell Ranger HDF5 (.h5)
    "h5ad",         # AnnData
    "rds",          # Seurat RDS
    "rdata",        # R RData / RDA
    "unknown",      # could not be determined
}

FORMAT_ANALYZER_SYSTEM_PROMPT = ChatPromptTemplate.from_template("""
You are an expert bioinformatician helping to configure an automated single-cell data
conversion pipeline.

You will be given:
1. Metadata about a dataset (species, technology, format hint from an earlier analysis step).
2. The list of files that were actually downloaded, with their sizes.
3. Optional: the first few lines of small text files.

Your task is to decide the **exact conversion strategy** for this dataset:

- **data_type** — the format key to pass to the converter. Must be one of:
    "10x"        → Cell Ranger output directory (has matrix.mtx.gz + barcodes.tsv.gz + features.tsv.gz)
    "10x_matrix" → single MTX file (not a full 10x directory)
    "csv"        → comma-separated count matrix
    "tsv" / "txt" / "csv.gz" / "tsv.gz" → tab- or comma-separated variants
    "mtx"        → plain MTX (not 10x)
    "h5"         → Cell Ranger HDF5 (.h5, not .h5ad)
    "h5ad"       → AnnData H5AD
    "rds"        → Seurat RDS object
    "rdata"      → R RData / RDA file
    "unknown"    → cannot be determined

- **primary_file** — the filename of the main count matrix to convert (not metadata CSVs).
  For a 10x directory, set this to the directory name that holds matrix.mtx.gz +
  barcodes.tsv.gz + features.tsv.gz. Archives (e.g. *_RAW.tar) are already
  unpacked for you — point primary_file at the extracted file/directory, never at
  the .tar/.zip itself.

- **species** — confirm or correct the species ("Human", "Mouse", or "Other").

- **gene_mapping_needed** — true if the gene IDs in the file are Ensembl IDs (ENSG…/ENSMUSG…)
  and need to be mapped to gene symbols. False if already gene symbols.

- **normalization_status** — your best assessment based on filenames, sizes, and any text
  content: "raw_counts", "normalized", or "unknown".

- **requires_r_extraction** — true if the file is a Seurat RDS and Python cannot read it
  directly (i.e., rpy2 / R is needed). This is usually true for .rds and .rdata files.

- **special_handling** — free-text note about any non-standard extraction step needed
  (e.g., "extract RAW.tar first", "counts slot inside Seurat object", "need rpy2").
  Set to null if no special handling is needed.

Key heuristics:
- `.rds` or `.rds.gz` files → data_type = "rds", requires_r_extraction = true
- `.rdata` / `.rda` files → data_type = "rdata", requires_r_extraction = true
- `matrix.mtx.gz` + `barcodes.tsv.gz` + `features.tsv.gz` in the same directory → data_type = "10x"
- Cell Ranger HDF5 (`*.h5`, e.g. `filtered_feature_bc_matrix.h5`, `raw_feature_bc_matrix.h5`,
  `*_raw_gene_bc_matrices_h5.h5`) → data_type = "h5"
- `*_RAW.tar` is auto-extracted: ignore the archive itself and classify by its
  EXTRACTED contents — e.g. many per-sample `*.h5` → "h5"; an mtx triplet dir → "10x".
- Multi-sample: when several per-sample matrix files of the same format are present
  (one `.h5` per GEO sample), set primary_file to the extracted DIRECTORY that holds
  them — the converter merges every sample into one h5ad automatically.
- `.h5ad` → data_type = "h5ad", requires_r_extraction = false
- Filename containing "processed", "integrated", "annotated", "normalized" → normalization_status = "normalized"
- Filename containing "raw", "counts", "count_matrix" → normalization_status = "raw_counts"

---

**Dataset metadata (from manifest):**
- dataset_id        : {dataset_id}
- species hint      : {species}
- technology        : {technology}
- confirmed_format  : {confirmed_format}
- raw_data_available: {raw_data_available}
- download_notes    : {download_notes}

**Downloaded files:**
{file_listing}

**Text file previews (if any):**
{text_previews}
""")


class ConversionConfigResult(BaseModel):
    reasoning_process: str = Field(
        description="Step-by-step reasoning for the chosen conversion strategy"
    )
    data_type: str = Field(
        description="Format key for SingleCellConverter: '10x', '10x_matrix', 'csv', 'tsv', "
                    "'csv.gz', 'tsv.gz', 'txt', 'mtx', 'h5', 'h5ad', 'rds', 'rdata', or 'unknown'"
    )
    primary_file: str = Field(
        description="Filename of the main count matrix to convert"
    )
    species: str = Field(
        description="Confirmed species: 'Human', 'Mouse', or 'Other'"
    )
    gene_mapping_needed: bool = Field(
        description="True if Ensembl gene IDs must be mapped to gene symbols"
    )
    normalization_status: str = Field(
        description="'raw_counts', 'normalized', or 'unknown'"
    )
    requires_r_extraction: bool = Field(
        description="True if rpy2/R is required to read this file format"
    )
    special_handling: Optional[str] = Field(
        default=None,
        description="Non-standard extraction note, or null if none"
    )


class FormatAnalyzerStep:
    """
    Pipeline 2, Step B1: determine the exact conversion config from downloaded files.

    Input  : data/1.manifest/{dataset_id}.json
             data/2.raw/{dataset_id}/download_status.json
    Output : data/2.raw/{dataset_id}/conversion_config.json
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

    def analyze(self, dataset_id: str) -> Optional[dict]:
        """
        Produce a conversion_config.json for dataset_id.

        Returns the config dict, or None on failure.
        Skips the LLM call if the config already exists.
        """
        self.last_token_usage = None
        config_path = self._config_path(dataset_id)
        if os.path.exists(config_path):
            logger.info("FormatAnalyzerStep: %s already analyzed, loading from disk", dataset_id)
            with open(config_path) as f:
                return json.load(f)

        manifest = self._load_manifest(dataset_id)
        if manifest is None:
            logger.error("FormatAnalyzerStep: manifest not found for %s", dataset_id)
            return None

        download_status = self._load_download_status(dataset_id)
        if download_status is None:
            logger.error("FormatAnalyzerStep: download_status not found for %s", dataset_id)
            return None

        file_listing = self._build_file_listing(download_status)
        text_previews = self._build_text_previews(dataset_id, download_status)

        result = self._run_llm(manifest, file_listing, text_previews)
        if result is None:
            logger.error("FormatAnalyzerStep: LLM returned None for %s", dataset_id)
            return None

        config = self._build_config(dataset_id, manifest, result)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(
            "FormatAnalyzerStep: wrote %s (data_type=%s, species=%s, r_extraction=%s)",
            config_path, config["data_type"], config["species"], config["requires_r_extraction"],
        )
        return config

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _manifest_path(self, dataset_id: str) -> str:
        return os.path.join(self.data_folder, MANIFEST_SUBDIR, f"{dataset_id}.json")

    def _status_path(self, dataset_id: str) -> str:
        return os.path.join(self.data_folder, RAW_SUBDIR, dataset_id, DOWNLOAD_STATUS_FILENAME)

    def _config_path(self, dataset_id: str) -> str:
        return os.path.join(self.data_folder, RAW_SUBDIR, dataset_id, CONVERSION_CONFIG_FILENAME)

    def _load_manifest(self, dataset_id: str) -> Optional[dict]:
        path = self._manifest_path(dataset_id)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def _load_download_status(self, dataset_id: str) -> Optional[dict]:
        path = self._status_path(dataset_id)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _build_file_listing(download_status: dict) -> str:
        lines = []
        for rec in download_status.get("files", []):
            size_mb = (rec.get("size_bytes") or 0) / 1024 / 1024
            status = rec.get("status", "unknown")
            lines.append(
                f"  [{status}] {rec['filename']}  ({size_mb:.1f} MB)"
            )
            # Archive members are unpacked by the downloader — list them so the
            # LLM can choose a real primary_file (e.g. the extracted 10x dir).
            extracted = rec.get("extracted_files")
            if extracted:
                dest = rec.get("extracted_to", "")
                for member in extracted:
                    rel = os.path.join(dest, member) if dest else member
                    lines.append(f"      └─ {rel}")
        return "\n".join(lines) if lines else "  (no files)"

    def _build_text_previews(self, dataset_id: str, download_status: dict) -> str:
        raw_dir = os.path.join(self.data_folder, RAW_SUBDIR, dataset_id)

        # (display_name, local_path) for each downloaded file and each extracted
        # archive member, so previews cover the real contents (e.g. features.tsv.gz).
        candidates: list[tuple[str, str]] = []
        for rec in download_status.get("files", []):
            if rec.get("status") not in ("success", "skipped"):
                continue
            candidates.append((rec["filename"], os.path.join(raw_dir, rec["filename"])))
            dest = rec.get("extracted_to", "")
            for member in rec.get("extracted_files", []) or []:
                rel = os.path.join(dest, member) if dest else member
                candidates.append((rel, os.path.join(raw_dir, rel)))

        previews = []
        for display_name, local_path in candidates:
            base = display_name.lower()
            if not any(base.endswith(ext) for ext in TEXT_EXTENSIONS):
                continue
            if not os.path.exists(local_path) or os.path.getsize(local_path) > TEXT_SNIFF_LIMIT:
                continue
            preview = self._peek_text(local_path)
            if preview is not None:
                previews.append(f"--- {display_name} (first 10 lines) ---\n{preview}")
        return "\n\n".join(previews) if previews else "(no small text files available)"

    @staticmethod
    def _peek_text(local_path: str) -> Optional[str]:
        try:
            import gzip
            opener = gzip.open if local_path.endswith(".gz") else open
            with opener(local_path, "rt", errors="replace") as fh:
                return "".join(fh.readlines()[:10])
        except Exception as exc:
            logger.debug("Could not peek %s: %s", local_path, exc)
            return None

    def _run_llm(
        self, manifest: dict, file_listing: str, text_previews: str
    ) -> Optional[ConversionConfigResult]:
        system_prompt = FORMAT_ANALYZER_SYSTEM_PROMPT.format(
            dataset_id=manifest.get("dataset_id", "N/A"),
            species=manifest.get("species", manifest.get("repository", "unknown")),
            technology=manifest.get("technology", "unknown"),
            confirmed_format=manifest.get("confirmed_format", "unknown"),
            raw_data_available=manifest.get("raw_data_available", "unknown"),
            download_notes=manifest.get("download_notes") or "none",
            file_listing=file_listing,
            text_previews=text_previews,
        )
        agent = CommonAgent(llm=self.llm)
        res, _, token_usage, _ = agent.go(
            system_prompt=system_prompt,
            instruction_prompt=(
                "Analyse the downloaded files and determine the best conversion strategy. "
                "Reason step by step, then fill in the structured output."
            ),
            schema=ConversionConfigResult,
        )
        self.last_token_usage = token_usage
        return res

    def _build_config(
        self, dataset_id: str, manifest: dict, result: ConversionConfigResult
    ) -> dict:
        # Normalise data_type to a known value
        data_type = result.data_type.lower().strip()
        if data_type not in VALID_DATA_TYPES:
            logger.warning(
                "FormatAnalyzerStep: unknown data_type '%s' for %s — keeping as-is",
                data_type, dataset_id,
            )

        # Repair the LLM's primary_file into a real path on disk when possible,
        # so it points at the actual matrix (e.g. an extracted 10x directory)
        # rather than a description. Stored relative to the dataset's raw dir.
        primary_file = result.primary_file
        if not result.requires_r_extraction:
            dataset_dir = os.path.join(self.data_folder, RAW_SUBDIR, dataset_id)
            resolved = resolve_input_path(dataset_dir, data_type, result.primary_file)
            if resolved:
                rel = os.path.relpath(resolved, dataset_dir)
                if rel != primary_file:
                    logger.info(
                        "FormatAnalyzerStep: resolved primary_file for %s: %r → %r",
                        dataset_id, result.primary_file, rel,
                    )
                primary_file = rel

        return {
            "dataset_id": dataset_id,
            "pmid": manifest.get("pmid", ""),
            "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reasoning": result.reasoning_process,
            "data_type": data_type,
            "primary_file": primary_file,
            "species": result.species,
            "gene_mapping_needed": result.gene_mapping_needed,
            "normalization_status": result.normalization_status,
            "requires_r_extraction": result.requires_r_extraction,
            "special_handling": result.special_handling,
        }
