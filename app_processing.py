"""
Pipeline 2 — Data Processing CLI.

Takes one or more PMIDs accepted by Pipeline 1, fetches each paper's full text,
then runs the full data-processing chain (metadata → download → convert → QC →
doublet → annotate → label → clean) for every dataset found, and optionally
merges all cleaned datasets into a single atlas.

Examples
--------
    # Process two papers and merge into an atlas named "alzheimer"
    python app_processing.py -p 39578645 39607927 --atlas-name alzheimer

    # Process one paper, skip the atlas merge
    python app_processing.py -p 39578645 --skip-atlas

    # Re-run from already-extracted metadata (no full-text fetch needed —
    # idempotent steps load existing intermediate files from disk)
    python app_processing.py -p 39578645
"""

import argparse
import logging
import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_openai.chat_models import AzureChatOpenAI

from src.agents.agent_utils import increase_token_usage
from src.agents.constants import DEFAULT_TOKEN_USAGE
from src.log_utils import initialize_logger
from src.paper_query.pubmed_query import PubMedPaperRetriever
from src.workflow.processing_workflow import ProcessingWorkflow
from src.workflow.workflow_utils import convert_html_to_plaintext

load_dotenv()

logger = initialize_logger(
    log_file="app_processing.log",
    app_log_name="app_processing_logger",
    app_log_level=logging.INFO,
    log_entries={
        "src": logging.INFO,
    },
)


def get_azure_openai() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        api_key=os.environ.get("OPENAI_4O_API_KEY", None),
        azure_endpoint=os.environ.get("AZURE_OPENAI_4O_ENDPOINT", None),
        api_version=os.environ.get("OPENAI_4O_API_VERSION", None),
        azure_deployment=os.environ.get("OPENAI_4O_DEPLOYMENT_NAME", None),
        model=os.environ.get("OPENAI_4O_MODEL", None),
        max_retries=5,
        max_completion_tokens=int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", 16380)),
    )


g_token_usage = {**DEFAULT_TOKEN_USAGE}


def output_step(
    step_name: Optional[str] = None,
    step_description: Optional[str] = None,
    step_output: Optional[str] = None,
    step_reasoning_process: Optional[str] = None,
    token_usage: Optional[dict] = None,
):
    global g_token_usage
    if step_name is not None:
        logger.info("-" * 64)
        logger.info(step_name)
    if step_description is not None:
        logger.info(step_description)
    if token_usage is not None:
        logger.info(
            "step tokens: total=%d, prompt=%d, completion=%d",
            token_usage["total_tokens"],
            token_usage["prompt_tokens"],
            token_usage["completion_tokens"],
        )
        g_token_usage = increase_token_usage(g_token_usage, token_usage)
        logger.info(
            "overall tokens: total=%d, prompt=%d, completion=%d",
            g_token_usage["total_tokens"],
            g_token_usage["prompt_tokens"],
            g_token_usage["completion_tokens"],
        )
    if step_output is not None:
        logger.info(step_output)
    if step_reasoning_process is not None:
        logger.info("\n%s\n", step_reasoning_process)


def _fetch_paper_text(retriever: PubMedPaperRetriever, pmid: str) -> tuple[str, str]:
    """Return (title, full_text) for a PMID, or ("", "") if unavailable."""
    title, _abstract, _is_preprint = retriever.query_title_abstract_ispreprint(pmid)
    ok, html = retriever.query_full_text(pmid)
    full_text = convert_html_to_plaintext(html) if (ok and html) else ""
    return (title or ""), full_text


def _metadata_exists(data_folder: str, pmid: str) -> bool:
    return os.path.exists(os.path.join(data_folder, "0.metadata", f"{pmid}.json"))


def execute_processing(
    pmids: List[str],
    atlas_name: Optional[str],
    skip_atlas: bool,
    data_folder: Optional[str] = None,
) -> dict:
    data_folder = data_folder or os.getenv("DATA_FOLDER", ".")
    retriever = PubMedPaperRetriever()
    wf = ProcessingWorkflow(
        llm=get_azure_openai(),
        data_folder=data_folder,
        step_callback=output_step,
    )
    wf.compile()

    all_dataset_ids: List[str] = []
    paper_summaries = []

    for pmid in pmids:
        logger.info("=" * 64)
        logger.info("Processing PMID: %s", pmid)

        title, full_text = _fetch_paper_text(retriever, pmid)
        if not full_text and not _metadata_exists(data_folder, pmid):
            logger.warning(
                "PMID %s: no full text and no cached metadata — skipping", pmid
            )
            continue

        summary = wf.run_paper(pmid, title, full_text)
        all_dataset_ids.extend(summary["dataset_ids"])
        paper_summaries.append(summary)

        for ds in summary["datasets"]:
            logger.info(
                "  %s → status=%s%s",
                ds["dataset_id"],
                ds["status"],
                f" (stopped at {ds['stopped_at']})" if ds.get("stopped_at") else "",
            )

    atlas_result = None
    if not skip_atlas and all_dataset_ids:
        logger.info("=" * 64)
        logger.info("Merging %d dataset(s) into atlas '%s'", len(all_dataset_ids), atlas_name)
        atlas_result = wf.build_atlas(all_dataset_ids, atlas_name=atlas_name or "atlas")
        logger.info("Atlas result: %s", atlas_result)

    logger.info("=" * 64)
    logger.info("Done. PMIDs processed: %d, datasets discovered: %d",
                len(paper_summaries), len(all_dataset_ids))
    logger.info(
        "RUN TOTAL tokens: total=%d, prompt=%d, completion=%d",
        wf.token_usage["total_tokens"],
        wf.token_usage["prompt_tokens"],
        wf.token_usage["completion_tokens"],
    )

    return {
        "papers": paper_summaries,
        "dataset_ids": all_dataset_ids,
        "atlas": atlas_result,
        "token_usage": dict(wf.token_usage),
    }


def main(args: dict) -> None:
    execute_processing(
        pmids=args["pmid"],
        atlas_name=args["atlas_name"],
        skip_atlas=args["skip_atlas"],
        data_folder=args.get("data_folder"),
    )
    for handler in logger.handlers:
        handler.flush()
    logging.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline 2: process accepted PMIDs into single-cell atlases."
    )
    parser.add_argument(
        "-p", "--pmid", nargs="+", required=True,
        help="One or more PMIDs accepted by Pipeline 1.",
    )
    parser.add_argument(
        "--atlas-name", default="atlas",
        help="Name for the merged atlas h5ad (default: atlas).",
    )
    parser.add_argument(
        "--skip-atlas", action="store_true",
        help="Process datasets but skip the final atlas merge.",
    )
    parser.add_argument(
        "-o", "--data-folder", default=None,
        help="Override DATA_FOLDER root for all pipeline I/O.",
    )
    args = vars(parser.parse_args())
    main(args)
