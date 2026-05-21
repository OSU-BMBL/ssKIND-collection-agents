"""
Pipeline 2, Step C2 — QCReviewerStep

Input  : data/4.qc/{dataset_id}_qc_report.json  (from CountQCStep)
Output : data/4.qc/{dataset_id}_thresholds.json

The LLM receives the QC summary statistics and suggested default thresholds,
then decides whether to keep or adjust each threshold based on the observed
distribution of the data.
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

QC_SUBDIR = "4.qc"
QC_REPORT_FILENAME_TMPL = "{dataset_id}_qc_report.json"
THRESHOLDS_FILENAME_TMPL = "{dataset_id}_thresholds.json"


QC_REVIEWER_SYSTEM_PROMPT = ChatPromptTemplate.from_template("""
You are an expert bioinformatician reviewing quality-control metrics for a
single-cell RNA-seq or spatial transcriptomics dataset before filtering.

You will be given:
1. Summary statistics for the dataset (cells, genes, count distributions, MT%, ribo%).
2. Suggested default QC thresholds.

Your task is to decide the **final filtering thresholds** for this dataset, reasoning
from the observed distributions.

Guidelines:
- If the data looks clean (low MT%, reasonable count distributions), keep the defaults.
- Raise max_pct_mt slightly (e.g. to 10–15%) if the dataset has a high mean MT% that
  likely reflects cell biology (e.g. cardiac or muscle tissue) rather than damage.
- Raise max_total_counts if the data is clearly enriched for large / complex cells.
- Lower min_total_counts if counts are globally low (e.g. nucleus-seq / low-input).
- For non-human / non-mouse datasets, only min_genes, min_cells, and max_total_counts
  are available.
- Always apply min_genes >= 200 and min_cells >= 3.
- Set approved = true unless the data looks fundamentally broken (all zeros, extreme
  skew that cannot be rescued by threshold adjustment).

---

**Dataset:** {dataset_id}
**Species:** {species}
**PMID:** {pmid}

**QC summary (before any filtering):**
- Cells       : {n_cells}
- Genes       : {n_genes}
- total_counts: mean={tc_mean:.1f}, median={tc_median:.1f}, p5={tc_p5:.1f}, p95={tc_p95:.1f}
- n_genes/cell: mean={ng_mean:.1f}, median={ng_median:.1f}, p5={ng_p5:.1f}, p95={ng_p95:.1f}
{mt_ribo_section}

**Suggested default thresholds:**
{suggested_thresholds}
""")


class QCThresholdsResult(BaseModel):
    reasoning_process: str = Field(
        description="Step-by-step reasoning for chosen thresholds"
    )
    approved: bool = Field(
        description="True if the dataset is usable after filtering, False if fatally broken"
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Only set when approved=False; brief reason for rejection"
    )
    min_genes: int = Field(description="Minimum genes per cell (>= 200)")
    min_cells: int = Field(description="Minimum cells per gene (>= 3)")
    max_genes: Optional[int] = Field(
        default=None,
        description="Maximum genes per cell; null for non-human/mouse"
    )
    min_total_counts: Optional[int] = Field(
        default=None,
        description="Minimum total UMI counts per cell; null for non-human/mouse"
    )
    max_total_counts: int = Field(description="Maximum total counts per cell")
    max_pct_mt: Optional[float] = Field(
        default=None,
        description="Maximum mitochondrial % per cell; null for non-human/mouse"
    )
    max_pct_ribo: Optional[float] = Field(
        default=None,
        description="Maximum ribosomal % per cell; null for non-human/mouse"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Any non-standard observations about this dataset's QC"
    )


class QCReviewerStep:
    """
    Pipeline 2, Step C2: LLM-driven QC threshold review.

    Input  : data/4.qc/{dataset_id}_qc_report.json
    Output : data/4.qc/{dataset_id}_thresholds.json
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

    def review(self, dataset_id: str) -> Optional[dict]:
        """
        Review QC metrics and produce filtering thresholds.

        Returns the thresholds dict, or None on failure.
        Skips LLM call if thresholds already exist on disk.
        """
        self.last_token_usage = None
        thresholds_path = self._thresholds_path(dataset_id)
        if os.path.exists(thresholds_path):
            logger.info("QCReviewerStep: %s already reviewed, loading from disk", dataset_id)
            with open(thresholds_path) as f:
                return json.load(f)

        qc_report = self._load_report(dataset_id)
        if qc_report is None:
            logger.error("QCReviewerStep: QC report not found for %s", dataset_id)
            return None

        result = self._run_llm(dataset_id, qc_report)
        if result is None:
            logger.error("QCReviewerStep: LLM returned None for %s", dataset_id)
            return None

        thresholds = self._build_thresholds(dataset_id, qc_report, result)
        with open(thresholds_path, "w") as f:
            json.dump(thresholds, f, indent=2)
        logger.info(
            "QCReviewerStep: %s → approved=%s, min_genes=%d, max_pct_mt=%s",
            dataset_id, thresholds["approved"],
            thresholds["thresholds"]["min_genes"],
            thresholds["thresholds"].get("max_pct_mt"),
        )
        return thresholds

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _report_path(self, dataset_id: str) -> str:
        return os.path.join(
            self.data_folder, QC_SUBDIR,
            QC_REPORT_FILENAME_TMPL.format(dataset_id=dataset_id),
        )

    def _thresholds_path(self, dataset_id: str) -> str:
        return os.path.join(
            self.data_folder, QC_SUBDIR,
            THRESHOLDS_FILENAME_TMPL.format(dataset_id=dataset_id),
        )

    def _load_report(self, dataset_id: str) -> Optional[dict]:
        path = self._report_path(dataset_id)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def _run_llm(
        self, dataset_id: str, qc_report: dict
    ) -> Optional[QCThresholdsResult]:
        summary = qc_report.get("summary", {})
        tc = summary.get("total_counts") or {}
        ng = summary.get("n_genes_by_counts") or {}

        is_hm = "human" in (qc_report.get("species") or "").lower() or \
                "mouse" in (qc_report.get("species") or "").lower()

        if is_hm:
            mt_ribo_section = (
                f"- MT %     : mean={summary.get('mean_pct_mt', 0):.2f}%, "
                f"p5={summary.get('pct_mt', {}).get('p5', 0):.2f}%, "
                f"p95={summary.get('pct_mt', {}).get('p95', 0):.2f}%\n"
                f"- Ribo %   : mean={summary.get('mean_pct_ribo', 0):.2f}%, "
                f"p5={summary.get('pct_ribo', {}).get('p5', 0):.2f}%, "
                f"p95={summary.get('pct_ribo', {}).get('p95', 0):.2f}%"
            )
        else:
            mt_ribo_section = "(MT/ribo metrics not available for this species)"

        system_prompt = QC_REVIEWER_SYSTEM_PROMPT.format(
            dataset_id=dataset_id,
            species=qc_report.get("species", "unknown"),
            pmid=qc_report.get("pmid", ""),
            n_cells=summary.get("n_cells", 0),
            n_genes=summary.get("n_genes", 0),
            tc_mean=tc.get("mean", 0),
            tc_median=tc.get("median", 0),
            tc_p5=tc.get("p5", 0),
            tc_p95=tc.get("p95", 0),
            ng_mean=ng.get("mean", 0),
            ng_median=ng.get("median", 0),
            ng_p5=ng.get("p5", 0),
            ng_p95=ng.get("p95", 0),
            mt_ribo_section=mt_ribo_section,
            suggested_thresholds=json.dumps(
                qc_report.get("suggested_thresholds", {}), indent=2
            ),
        )
        agent = CommonAgent(llm=self.llm)
        res, _, token_usage, _ = agent.go(
            system_prompt=system_prompt,
            instruction_prompt=(
                "Review the QC metrics and decide the final filtering thresholds. "
                "Reason step by step, then fill in the structured output."
            ),
            schema=QCThresholdsResult,
        )
        self.last_token_usage = token_usage
        return res

    @staticmethod
    def _build_thresholds(
        dataset_id: str, qc_report: dict, result: QCThresholdsResult
    ) -> dict:
        return {
            "dataset_id": dataset_id,
            "pmid": qc_report.get("pmid", ""),
            "species": qc_report.get("species", ""),
            "reviewed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reasoning": result.reasoning_process,
            "approved": result.approved,
            "rejection_reason": result.rejection_reason,
            "thresholds": {
                "min_genes": result.min_genes,
                "min_cells": result.min_cells,
                "max_genes": result.max_genes,
                "min_total_counts": result.min_total_counts,
                "max_total_counts": result.max_total_counts,
                "max_pct_mt": result.max_pct_mt,
                "max_pct_ribo": result.max_pct_ribo,
            },
            "notes": result.notes,
        }
