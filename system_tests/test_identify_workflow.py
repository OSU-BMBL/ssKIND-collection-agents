import pytest

from src.agents.agent_utils import ResearchGoalEnum
from src.workflow.identify_workflow import IdentifyWorkflow, identify_workflow

@pytest.mark.skip()
def test_IdentifyWorkflow_sc_alzheimer(
    llm, 
    step_callback,
    sc_alzheimers_identify_original_instructions,
    sc_alzheimers_identify_relevant_instructions,
):
    # Initialize the IdentifyWorkflow with a mock LLM and step callback
    workflow = IdentifyWorkflow(
        llm=llm, 
        step_callback=step_callback,
        two_steps_agent=True,
    )
    
    # Compile the workflow
    workflow.compile()
    
    # Define a sample PMID and research goal
    pmid = "38566045"
    
    result = identify_workflow(
        wf=workflow,
        pmid=pmid,
        research_goal="Alzheimer_SingleCell",
        identify_original_instructions=sc_alzheimers_identify_original_instructions,
        identify_relevant_instructions=sc_alzheimers_identify_relevant_instructions,
    )
        
    # Check if the result is a boolean indicating relevance
    assert isinstance(result, bool)
    step_callback(step_output=f"{pmid} is {'relevant' if result else 'NOT relevant'}")

@pytest.mark.skip()
def test_IdentifyWorkflow_on_spatial(llm, step_callback):
    # Initialize the IdentifyWorkflow with a mock LLM and step callback
    workflow = IdentifyWorkflow(llm=llm, step_callback=step_callback)
    
    # Compile the workflow
    workflow.compile()
    
    # Define a sample PMID and research goal
    pmid = "40462993" # "39641382" # "39951525" # "38987616"
    research_goal = ResearchGoalEnum.SPATIAL
    
    # Execute the identify method
    result = workflow.identify(pmid=pmid, research_goal=research_goal)
    
    # Check if the result is a boolean indicating relevance
    assert isinstance(result, bool)

def test_IdentifyWorkflow_sc_parkinson(
    llm, 
    step_callback,
    sc_alzheimers_identify_original_instructions,
    sc_alzheimers_identify_relevant_instructions,
):
    # Initialize the IdentifyWorkflow with a mock LLM and step callback
    workflow = IdentifyWorkflow(
        llm=llm, 
        step_callback=step_callback,
        two_steps_agent=True,
    )
    
    # Compile the workflow
    workflow.compile()
    
    # Define a sample PMID and research goal
    pmid = "36117858"
    
    result = identify_workflow(
        wf=workflow,
        pmid=pmid,
        research_goal="Parkinson_SingleCell",
        identify_original_instructions=sc_alzheimers_identify_original_instructions,
        identify_relevant_instructions=sc_alzheimers_identify_relevant_instructions,
    )
        
    # Check if the result is a boolean indicating relevance
    assert isinstance(result, bool)
    step_callback(step_output=f"{pmid} is {'relevant' if result else 'NOT relevant'}")