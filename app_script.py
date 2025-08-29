import argparse
import os
from typing import Optional
from dotenv import load_dotenv
import logging
from langchain_openai.chat_models import AzureChatOpenAI

from src.agents.agent_utils import ResearchGoalEnum, increase_token_usage
from src.agents.constants import DEFAULT_TOKEN_USAGE

from src.config_utils import (
    read_config_identify_original_instructions, 
    read_config_identify_relevant_instructions, 
    read_config_query,
    read_config_scopes,
)
from src.paper_query.pubmed_query import (
    query_count,
    query_pmids,
)
from src.workflow.identify_workflow import IdentifyWorkflow, identify_workflow
from src.log_utils import initialize_logger

load_dotenv()

logger = initialize_logger(
    log_file="app.log",
    app_log_name="app_logger",
    app_log_level=logging.INFO,
    log_entries={
        "src": logging.INFO,
    }
)

def get_azure_openai():
    return AzureChatOpenAI(
        api_key=os.environ.get("OPENAI_4O_API_KEY", None),
        azure_endpoint=os.environ.get("AZURE_OPENAI_4O_ENDPOINT", None),
        api_version=os.environ.get("OPENAI_4O_API_VERSION", None),
        azure_deployment=os.environ.get("OPENAI_4O_DEPLOYMENT_NAME", None),
        model=os.environ.get("OPENAI_4O_MODEL", None),
        max_retries=5,
        # temperature=0.0,
        max_completion_tokens=int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", 4096)),
    )

def output_info(msg: str):
    logger.info(msg)

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
        output_info("=" * 64)
        output_info(step_name)
    if step_description is not None:
        output_info(step_description)
    if token_usage is not None:
        usage_str = f"step total tokens: {token_usage['total_tokens']}, step prompt tokens: {token_usage['prompt_tokens']}, step completion tokens: {token_usage['completion_tokens']}"
        output_info(usage_str)
        g_token_usage = increase_token_usage(g_token_usage, token_usage)
        usage_str = f"overall total tokens: {g_token_usage['total_tokens']}, overall prompt tokens: {g_token_usage['prompt_tokens']}, overall completion tokens: {g_token_usage['completion_tokens']}"
        output_info(usage_str)
    if step_reasoning_process is not None:
        output_info(f"\n\n{step_reasoning_process}\n\n")
    if step_output is not None:
        output_info(step_output)

def output_collect_result(
    scope: str,
    pmid: str, 
    relevant: bool
):
    with open("results.txt", "a") as f:
        if relevant:
            f.write(f"{pmid} is relevant to {scope}.\n")
        else:
            f.write(f"{pmid} is NOT relevant to {scope}.\n")

def execute_collection(
    scope: str,
    query: str,
    mindate: str,
    maxdate: str,
    identify_original_instructions: str,
    identify_relevant_instructions: str,
): 
    count = query_count(query, mindate, maxdate)
    logger.info(f"Total articles found: {count}")
    pmids = query_pmids(query, count, mindate, maxdate)
    all_pmids = []
    wf = IdentifyWorkflow(
        llm=get_azure_openai(),
        step_callback=output_step,
        two_steps_agent=True,
    )
    wf.compile()
    valid_pmids = []
    for pmid in pmids:
        all_pmids.append(pmid)
        logger.info(f"PMID: {pmid}")
        valid = identify_workflow(
            wf=wf,
            pmid=pmid,
            research_goal=scope,
            identify_original_instructions=identify_original_instructions,
            identify_relevant_instructions=identify_relevant_instructions,
        )
        if valid:
            valid_pmids.append(pmid)
            logger.info(f"PMID {pmid} is relevant to {scope}")
        else:
            logger.info(f"PMID {pmid} is NOT relevant to {scope}")
        output_collect_result(
            scope=scope, pmid=pmid, relevant=valid
        )
    
    logger.info("=" * 64)
    logger.info(f"Final Result: {scope}")
    logger.info(f"Query results number: {len(all_pmids)}, Total relevant PMIDs: {len(valid_pmids)}")
    logger.info(f"Relevant PMIDs: {valid_pmids}")
    
    return valid_pmids

def main_execute(scope: str):
    query, mindate, maxdate = read_config_query(scope) # '("Alzheimer") AND ("scRNA-seq" OR "single cell RNA sequencing"  OR "snRNA-seq" OR "single nucleus RNA sequencing")' # '(Alzheimer AND ("single cell" OR "single nucleus" OR "single-cell")) AND ("RNA sequencing" OR "RNA-seq" OR "single-cell RNA-seq")'
    identify_original_instructions = read_config_identify_original_instructions(scope)
    identify_relevant_instructions = read_config_identify_relevant_instructions(scope)
    valid_pmids = execute_collection(
        scope=scope,
        query=query,
        mindate=mindate,
        maxdate=maxdate,
        identify_original_instructions=identify_original_instructions,
        identify_relevant_instructions=identify_relevant_instructions,
    )
    
    return valid_pmids    


def main(scope):
    main_execute(scope)

    for handler in logger.handlers:
        handler.flush()
    logging.shutdown()

if __name__ == "__main__":
    entries = read_config_scopes()
    str_entries = str(entries)
    str_entries = str_entries[1:]
    str_entries = str_entries[:-1]
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scope", default="SC_Alzheimer", help=f"disease scope, like {str_entries}")
    args = parser.parse_args()
    args = vars(args)
    if not args["scope"] or args["scope"] not in entries:
        parser.print_usage()
    else:
        main(args["scope"])
