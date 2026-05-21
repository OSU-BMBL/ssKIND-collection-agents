"""
Pipeline 2, Step E1 — AnnotationConfigStep

Input  : data/2.raw/{dataset_id}/conversion_config.json
         data/4.qc/{dataset_id}_filter_result.json  (optional, for cell count)
Output : data/5.doublet/{dataset_id}_annotation_config.json

The LLM decides whether MapMyCells cell-type annotation should be run for
this dataset, which taxonomy to use, and any special considerations.
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

logger = logging.getLogger(__name__)

RAW_SUBDIR = "2.raw"
QC_SUBDIR = "4.qc"
DOUBLET_SUBDIR = "5.doublet"
CONVERSION_CONFIG_FILENAME = "conversion_config.json"
ANNOTATION_CONFIG_FILENAME = "{dataset_id}_annotation_config.json"

# Valid taxonomy choices passed to CellTypeAnnotationStep
VALID_TAXONOMIES = {"human_whole_brain", "mouse_whole_brain", "none"}


ANNOTATION_CONFIG_SYSTEM_PROMPT = ChatPromptTemplate.from_template("""
You are an expert bioinformatician deciding whether automated cell-type annotation
should be run on a single-cell dataset.

MapMyCells (Allen Institute) supports Human whole-brain and Mouse whole-brain
taxonomies. It is only applicable when:
- Species is Human or Mouse
- Data contains gene symbols (not just Ensembl IDs)
- Dataset has at least a few hundred cells after QC
- Data is single-cell or single-nucleus RNA-seq (not bulk, not spatial-only)

Your task:
1. Decide whether to run annotation (annotate = true/false).
2. If annotating, choose the taxonomy:
   - "human_whole_brain"  for Human scRNA-seq / snRNA-seq
   - "mouse_whole_brain"  for Mouse scRNA-seq / snRNA-seq
   - "none"               if annotation should be skipped
3. Note any concerns (e.g., spatial data where annotation is less reliable).

---

**Dataset:** {dataset_id}
**Species:** {species}
**Technology:** {technology}
**Data type:** {data_type}
**Gene mapping done:** {gene_mapping_needed}
**Normalization status:** {normalization_status}
**Cells after QC:** {n_cells_after_qc}
""")


class AnnotationConfigResult(BaseModel):
    reasoning_process: str = Field(
        description="Step-by-step reasoning for the annotation decision"
    )
    annotate: bool = Field(
        description="True if MapMyCells annotation should be run"
    )
    taxonomy: str = Field(
        description="Taxonomy to use: 'human_whole_brain', 'mouse_whole_brain', or 'none'"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Any special considerations or caveats"
    )


class AnnotationConfigStep:
    """
    Pipeline 2, Step E1: LLM decides whether and how to annotate cell types.

    Input  : data/2.raw/{dataset_id}/conversion_config.json
    Output : data/5.doublet/{dataset_id}_annotation_config.json
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

    def configure(self, dataset_id: str) -> Optional[dict]:
        """
        Produce an annotation_config.json for dataset_id.

        Returns the config dict, or None on failure.
        Skips LLM if the config already exists.
        """
        self.last_token_usage = None
        config_path = self._config_path(dataset_id)
        if os.path.exists(config_path):
            logger.info("AnnotationConfigStep: %s already configured, loading from disk", dataset_id)
            with open(config_path) as f:
                return json.load(f)

        conversion_config = self._load_conversion_config(dataset_id)
        if conversion_config is None:
            logger.error("AnnotationConfigStep: conversion_config.json not found for %s", dataset_id)
            return None

        n_cells_after_qc = self._load_n_cells_after_qc(dataset_id)

        result = self._run_llm(dataset_id, conversion_config, n_cells_after_qc)
        if result is None:
            logger.error("AnnotationConfigStep: LLM returned None for %s", dataset_id)
            return None

        taxonomy = result.taxonomy.lower().strip()
        if taxonomy not in VALID_TAXONOMIES:
            logger.warning(
                "AnnotationConfigStep: unrecognised taxonomy '%s' for %s — defaulting to 'none'",
                taxonomy, dataset_id,
            )
            taxonomy = "none"

        config = {
            "dataset_id": dataset_id,
            "pmid": conversion_config.get("pmid", ""),
            "configured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reasoning": result.reasoning_process,
            "annotate": result.annotate,
            "taxonomy": taxonomy,
            "notes": result.notes,
        }
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(
            "AnnotationConfigStep: %s → annotate=%s, taxonomy=%s",
            dataset_id, config["annotate"], config["taxonomy"],
        )
        return config

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _config_path(self, dataset_id: str) -> str:
        return os.path.join(
            self.data_folder, DOUBLET_SUBDIR,
            ANNOTATION_CONFIG_FILENAME.format(dataset_id=dataset_id),
        )

    def _load_conversion_config(self, dataset_id: str) -> Optional[dict]:
        path = os.path.join(
            self.data_folder, RAW_SUBDIR, dataset_id, CONVERSION_CONFIG_FILENAME
        )
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def _load_n_cells_after_qc(self, dataset_id: str) -> Optional[int]:
        path = os.path.join(
            self.data_folder, QC_SUBDIR, f"{dataset_id}_filter_result.json"
        )
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return data.get("n_cells_after")
        except Exception:
            return None

    def _run_llm(
        self,
        dataset_id: str,
        conversion_config: dict,
        n_cells_after_qc: Optional[int],
    ) -> Optional[AnnotationConfigResult]:
        system_prompt = ANNOTATION_CONFIG_SYSTEM_PROMPT.format(
            dataset_id=dataset_id,
            species=conversion_config.get("species", "unknown"),
            technology=conversion_config.get("data_type", "unknown"),
            data_type=conversion_config.get("data_type", "unknown"),
            gene_mapping_needed=conversion_config.get("gene_mapping_needed", False),
            normalization_status=conversion_config.get("normalization_status", "unknown"),
            n_cells_after_qc=n_cells_after_qc if n_cells_after_qc is not None else "unknown",
        )
        agent = CommonAgent(llm=self.llm)
        res, _, token_usage, _ = agent.go(
            system_prompt=system_prompt,
            instruction_prompt=(
                "Decide whether to run MapMyCells annotation for this dataset. "
                "Reason step by step, then fill in the structured output."
            ),
            schema=AnnotationConfigResult,
        )
        self.last_token_usage = token_usage
        return res
