
import pytest
import logging
from src.agents.identify_relevant_step import IdentifyRelevanceStep
from src.agents.identify_original_step import IdentifyOriginalDataStep
from src.agents.agent_utils import IdentifyState, ResearchGoalEnum
from src.paper_query.pubmed_query import query_title_abstract_ispreprint
from src.workflow.workflow_utils import obtain_full_text

logger = logging.getLogger(__name__)

@pytest.mark.skip()
def test_IdentifyRelevanceStep(llm, step_callback):
    # Create a mock state with title and abstract
    state = {
        "research_goal": ResearchGoalEnum.ALZHEIMERS,
        "title": "Single cell transcriptomes and multiscale networks from persons with and without Alzheimer's disease",
        "abstract": "The emergence of single nucleus RNA sequencing (snRNA-seq) offers to revolutionize the study of Alzheimer's disease (AD). Integration with complementary multiomics data such as genetics, proteomics and clinical data provides powerful opportunities to link cell subpopulations and molecular networks with a broader disease-relevant context. We report snRNA-seq profiles from superior frontal gyrus samples from 101 well characterized subjects from the Banner Brain and Body Donation Program in combination with whole genome sequences. We report findings that link common AD risk variants with CR1 expression in oligodendrocytes as well as alterations in hematological parameters. We observed an AD-associated CD83(+) microglial subtype with unique molecular networks and which is associated with immunoglobulin IgG4 production in the transverse colon. Our major observations were replicated in two additional, independent snRNA-seq data sets. These findings illustrate the power of multi-tissue molecular profiling to contextualize snRNA-seq brain transcriptomics and reveal disease biology.",
        "step_output_callback": step_callback,
    }
    
    # Initialize the IdentifyRelevanceStep with a mock LLM
    step = IdentifyRelevanceStep(llm)
    
    # Execute the step
    result_state: dict | None = step.execute(state)
    assert result_state is not None
    
    # Check if the result contains the expected keys
    assert "relevance" in result_state
    assert isinstance(result_state["relevance"], bool)  # Should be a boolean value indicating relevance

def test_IdentifyRelevanceStep_on_paper(llm, step_callback):
    pmid = "40448997"
    title, abstract, is_preprint = query_title_abstract_ispreprint(pmid)
    # Create a mock state with title and abstract
    state = {
        "research_goal": ResearchGoalEnum.ALZHEIMERS,
        "title": title,
        "abstract": abstract,
        "step_output_callback": step_callback,
    }
    
    # Initialize the IdentifyRelevanceStep with a mock LLM
    step = IdentifyRelevanceStep(llm, two_steps_agent=True)
    
    # Execute the step
    result_state: IdentifyState | None = step.execute(state)
    assert result_state is not None
    
    # Check if the result contains the expected keys
    assert "relevant" in result_state
    assert isinstance(result_state["relevant"], bool)  # Should be a boolean value indicating relevance

    logger.info(f"PMID: {pmid}, Relevant: {result_state['relevant']}")