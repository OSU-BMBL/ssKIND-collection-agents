"""
Pipeline 2 — Data Processing Orchestrator.

Wires the file-based, idempotent processing steps (A0 → H) into a single
workflow. Communication between steps is via intermediate files on disk under
``$DATA_FOLDER`` — the LangGraph state only carries routing/status information,
never the data itself.

Two scopes:

* **Per-dataset chain** (A1 → G): a LangGraph ``StateGraph`` that runs the
  download → convert → QC → doublet → annotate → label → clean sequence for one
  ``dataset_id``. Conditional edges short-circuit to ``END`` as soon as a step
  fails or flags a stop condition (e.g. an RDS file that needs R extraction, or
  a QC filter that removes every cell).
* **Atlas merge** (H): runs once across every dataset that produced a cleaned
  h5ad.

A0 (metadata extraction) runs once per PMID and fans out into N ``dataset_id``s.

Typical use::

    wf = ProcessingWorkflow(llm=..., data_folder=...)
    wf.compile()
    summary = wf.run_paper(pmid, title, full_text)   # A0 + per-dataset chain
    wf.build_atlas(all_dataset_ids, atlas_name="alzheimer")  # H
"""

from __future__ import annotations

import json
import logging
import os
from typing import Callable, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_openai.chat_models.base import BaseChatOpenAI

from ..agents.agent_utils import increase_token_usage
from ..agents.constants import DEFAULT_TOKEN_USAGE
from ..agents.metadata_extractor_step import (
    MetadataExtractorStep,
    METADATA_EXTRACTION_OUTPUT_SUBDIR,
)
from ..agents.repository_analyst_step import RepositoryAnalystStep
from ..agents.format_analyzer_step import FormatAnalyzerStep
from ..agents.qc_reviewer_step import QCReviewerStep
from ..agents.annotation_config_step import AnnotationConfigStep
from ..data_processing.data_downloader import DataDownloaderStep
from ..data_processing.format_converter import FormatConverterStep
from ..data_processing.count_qc_step import CountQCStep
from ..data_processing.qc_filter_step import QCFilterStep
from ..data_processing.doublet_detection_step import DoubletDetectionStep
from ..data_processing.cell_type_annotation_step import CellTypeAnnotationStep
from ..data_processing.label_merger_step import LabelMergerStep
from ..data_processing.atlas_cleaner_step import AtlasCleanerStep
from ..data_processing.atlas_merger_step import AtlasMergerStep

logger = logging.getLogger(__name__)


class ProcessingState(TypedDict, total=False):
    """Routing state for the per-dataset chain. Data lives on disk, not here."""

    pmid: str
    dataset_id: str
    proceed: bool
    status: str  # "completed" | "stopped"
    stopped_at: Optional[str]
    results: dict  # step_name -> step result dict
    step_output_callback: Optional[Callable]


class ProcessingWorkflow:
    """Orchestrates Pipeline 2 over the file-based processing steps."""

    # Per-dataset node order. Each entry is (node_name, node_method_attr).
    _CHAIN = [
        "manifest",
        "download",
        "format_analyze",
        "convert",
        "count_qc",
        "qc_review",
        "qc_filter",
        "doublet",
        "annotation_config",
        "cell_type_annotate",
        "label_merge",
        "atlas_clean",
    ]

    def __init__(
        self,
        llm: BaseChatOpenAI,
        data_folder: Optional[str] = None,
        step_callback: Optional[Callable] = None,
        ncbi_email: Optional[str] = None,
        human_gene_mart_path: Optional[str] = None,
        mouse_gene_mart_path: Optional[str] = None,
        human_taxonomy_path: Optional[str] = None,
        mouse_taxonomy_path: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.data_folder = data_folder or os.getenv("DATA_FOLDER", ".")
        self.step_callback = step_callback

        # A0 — metadata extraction (per PMID)
        self.metadata_extractor = MetadataExtractorStep(llm, self.data_folder)
        # A1..G — per-dataset chain
        self.repository_analyst = RepositoryAnalystStep(llm, self.data_folder, ncbi_email)
        self.data_downloader = DataDownloaderStep(self.data_folder)
        self.format_analyzer = FormatAnalyzerStep(llm, self.data_folder)
        self.format_converter = FormatConverterStep(
            self.data_folder, human_gene_mart_path, mouse_gene_mart_path
        )
        self.count_qc = CountQCStep(self.data_folder)
        self.qc_reviewer = QCReviewerStep(llm, self.data_folder)
        self.qc_filter = QCFilterStep(self.data_folder)
        self.doublet = DoubletDetectionStep(self.data_folder)
        self.annotation_config = AnnotationConfigStep(llm, self.data_folder)
        self.cell_type_annotation = CellTypeAnnotationStep(
            self.data_folder, human_taxonomy_path, mouse_taxonomy_path
        )
        self.label_merger = LabelMergerStep(self.data_folder)
        self.atlas_cleaner = AtlasCleanerStep(self.data_folder)
        # H — atlas merge (across datasets)
        self.atlas_merger = AtlasMergerStep(self.data_folder)

        self.graph = None

        # Running token-usage total across the whole run (the LLM steps are
        # the only token consumers). Reset with reset_token_usage().
        self.token_usage = {**DEFAULT_TOKEN_USAGE}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def compile(self) -> None:
        """Build the per-dataset LangGraph chain with short-circuit routing."""
        node_fns = {
            "manifest": self._node_manifest,
            "download": self._node_download,
            "format_analyze": self._node_format_analyze,
            "convert": self._node_convert,
            "count_qc": self._node_count_qc,
            "qc_review": self._node_qc_review,
            "qc_filter": self._node_qc_filter,
            "doublet": self._node_doublet,
            "annotation_config": self._node_annotation_config,
            "cell_type_annotate": self._node_cell_type_annotate,
            "label_merge": self._node_label_merge,
            "atlas_clean": self._node_atlas_clean,
        }

        graph = StateGraph(ProcessingState)
        for name in self._CHAIN:
            graph.add_node(name, node_fns[name])

        graph.add_edge(START, self._CHAIN[0])
        for i, name in enumerate(self._CHAIN):
            nxt = self._CHAIN[i + 1] if i + 1 < len(self._CHAIN) else END
            graph.add_conditional_edges(name, self._route, {"continue": nxt, "stop": END})

        self.graph = graph.compile()

    @staticmethod
    def _route(state: ProcessingState) -> str:
        return "continue" if state.get("proceed") else "stop"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_token_usage(self) -> None:
        """Zero the running token-usage total (call between independent runs)."""
        self.token_usage = {**DEFAULT_TOKEN_USAGE}

    def extract_metadata(self, pmid: str, title: str, full_text: str) -> List[str]:
        """A0: extract dataset metadata for a PMID. Returns the dataset_ids."""
        datasets = self.metadata_extractor.extract(pmid, title, full_text)
        self._accumulate(self.metadata_extractor)
        if not datasets:
            return []
        return [d["dataset_id"] for d in datasets if d.get("dataset_id")]

    def process_dataset(self, pmid: str, dataset_id: str) -> dict:
        """Run the A1→G chain for one dataset. Returns a summary dict."""
        if self.graph is None:
            self.compile()

        state: ProcessingState = {
            "pmid": pmid,
            "dataset_id": dataset_id,
            "proceed": True,
            "status": "completed",
            "stopped_at": None,
            "results": {},
            "step_output_callback": self.step_callback,
        }

        final: Optional[ProcessingState] = None
        for final in self.graph.stream(
            state, stream_mode="values", config={"recursion_limit": 100}
        ):
            continue

        final = final or state
        return {
            "pmid": pmid,
            "dataset_id": dataset_id,
            "status": final.get("status"),
            "stopped_at": final.get("stopped_at"),
            "results": final.get("results", {}),
            "token_usage": dict(self.token_usage),  # running total snapshot
        }

    def run_paper(self, pmid: str, title: str, full_text: str) -> dict:
        """A0 + per-dataset chain for every dataset in one paper."""
        dataset_ids = self.extract_metadata(pmid, title, full_text)
        summaries = [self.process_dataset(pmid, did) for did in dataset_ids]
        return {
            "pmid": pmid,
            "dataset_ids": dataset_ids,
            "datasets": summaries,
            "token_usage": dict(self.token_usage),  # running total snapshot
        }

    def build_atlas(self, dataset_ids: List[str], atlas_name: str = "atlas") -> Optional[dict]:
        """H: merge every cleaned dataset into a single atlas h5ad."""
        return self.atlas_merger.merge(dataset_ids, atlas_name=atlas_name)

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    def _node_manifest(self, state: ProcessingState) -> dict:
        dataset = self._dataset_entry(state["pmid"], state["dataset_id"])
        if dataset is None:
            return self._record(
                state,
                "manifest",
                {"status": "failed", "message": "dataset entry not found in metadata"},
                proceed=False,
            )
        result = self.repository_analyst.analyze(dataset)
        usage = self._accumulate(self.repository_analyst)
        ok = bool(result and result.get("files"))
        return self._record(state, "manifest", result, ok, token_usage=usage)

    def _node_download(self, state: ProcessingState) -> dict:
        result = self.data_downloader.download(state["dataset_id"])
        ok = bool(
            result
            and any(f.get("status") == "success" for f in result.get("files", []))
        )
        return self._record(state, "download", result, ok)

    def _node_format_analyze(self, state: ProcessingState) -> dict:
        result = self.format_analyzer.analyze(state["dataset_id"])
        usage = self._accumulate(self.format_analyzer)
        return self._record(state, "format_analyze", result, result is not None, token_usage=usage)

    def _node_convert(self, state: ProcessingState) -> dict:
        result = self.format_converter.convert(state["dataset_id"])
        # requires_r_extraction / failed → Python pipeline cannot continue
        ok = bool(result and result.get("status") in ("success", "skipped"))
        return self._record(state, "convert", result, ok)

    def _node_count_qc(self, state: ProcessingState) -> dict:
        result = self.count_qc.run(state["dataset_id"])
        ok = bool(result and result.get("status") != "failed")
        return self._record(state, "count_qc", result, ok)

    def _node_qc_review(self, state: ProcessingState) -> dict:
        result = self.qc_reviewer.review(state["dataset_id"])
        usage = self._accumulate(self.qc_reviewer)
        return self._record(state, "qc_review", result, result is not None, token_usage=usage)

    def _node_qc_filter(self, state: ProcessingState) -> dict:
        result = self.qc_filter.filter(state["dataset_id"])
        ok = bool(
            result
            and result.get("status") == "success"
            and result.get("n_cells_after", 0) > 0
        )
        return self._record(state, "qc_filter", result, ok)

    def _node_doublet(self, state: ProcessingState) -> dict:
        result = self.doublet.run(state["dataset_id"])
        ok = bool(result and result.get("status") == "success")
        return self._record(state, "doublet", result, ok)

    def _node_annotation_config(self, state: ProcessingState) -> dict:
        result = self.annotation_config.configure(state["dataset_id"])
        usage = self._accumulate(self.annotation_config)
        return self._record(state, "annotation_config", result, result is not None, token_usage=usage)

    def _node_cell_type_annotate(self, state: ProcessingState) -> dict:
        # Annotation is best-effort: success/skipped/failed all let the chain
        # continue — LabelMergerStep falls back to a "no labels" merge.
        result = self.cell_type_annotation.annotate(state["dataset_id"])
        return self._record(state, "cell_type_annotate", result, result is not None)

    def _node_label_merge(self, state: ProcessingState) -> dict:
        result = self.label_merger.merge(state["dataset_id"])
        ok = bool(result and result.get("status") in ("success", "success_no_labels"))
        return self._record(state, "label_merge", result, ok)

    def _node_atlas_clean(self, state: ProcessingState) -> dict:
        # Terminal node — routing irrelevant, but record success/failure.
        result = self.atlas_cleaner.clean(state["dataset_id"])
        ok = bool(result and result.get("status") in ("success", "skipped"))
        return self._record(state, "atlas_clean", result, ok)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _accumulate(self, step) -> Optional[dict]:
        """Fold an LLM step's last token usage into the running total.

        Returns the per-step usage (or None for cache hits / non-LLM steps).
        Relies on each LLM step setting ``last_token_usage`` to None at the top
        of its public method, so a skip path contributes nothing.
        """
        usage = getattr(step, "last_token_usage", None)
        if usage:
            self.token_usage = increase_token_usage(self.token_usage, usage)
        return usage

    def _record(
        self,
        state: ProcessingState,
        step_name: str,
        result,
        proceed: bool,
        token_usage: Optional[dict] = None,
    ) -> dict:
        """Store a step result, emit a callback, and set routing flags."""
        self._emit(step_name, result, state, token_usage)
        results = dict(state.get("results", {}))
        results[step_name] = result
        updates: dict = {"results": results, "proceed": proceed}
        if not proceed:
            updates["stopped_at"] = step_name
            updates["status"] = "stopped"
        return updates

    def _emit(
        self,
        step_name: str,
        result,
        state: ProcessingState,
        token_usage: Optional[dict] = None,
    ) -> None:
        cb = state.get("step_output_callback")
        if cb is None:
            return
        status = result.get("status") if isinstance(result, dict) else None
        try:
            cb(
                step_name=f"Pipeline2: {step_name} [{state.get('dataset_id')}]",
                step_output=(status or json.dumps(result, default=str)[:500]),
                token_usage=token_usage,
            )
        except Exception:  # callback must never break the pipeline
            logger.debug("step_callback raised for %s", step_name, exc_info=True)

    def _dataset_entry(self, pmid: str, dataset_id: str) -> Optional[dict]:
        """Look up one dataset's metadata dict from 0.metadata/{pmid}.json."""
        path = os.path.join(
            self.data_folder, METADATA_EXTRACTION_OUTPUT_SUBDIR, f"{pmid}.json"
        )
        if not os.path.exists(path):
            return None
        with open(path) as f:
            meta = json.load(f)
        for d in meta.get("datasets", []):
            if d.get("dataset_id") == dataset_id:
                return d
        return None
