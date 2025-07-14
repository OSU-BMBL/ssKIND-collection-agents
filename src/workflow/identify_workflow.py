
from typing import Callable, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai.chat_models.base import BaseChatOpenAI

from ..paper_query.pubmed_query import (
    query_title_abstract_ispreprint,
    query_full_text,
)
from ..agents.identify_relevant_step import (
    IdentifyRelevanceStep,
)
from ..agents.identify_original_step import (
    IdentifyOriginalDataStep,
)
from ..agents.agent_utils import IdentifyState, ResearchGoalEnum

from .workflow_utils import convert_html_to_plaintext
from ..paper_query.pubmed_query import PubMedPaperRetriever

class IdentifyWorkflow:
    def __init__(
        self, 
        llm: BaseChatOpenAI, 
        step_callback: Optional[Callable] = None,
        two_steps_agent: bool = False,
    ):
        self.llm = llm
        self.steps = []
        self.step_callback = step_callback
        self.paper_retriever = PubMedPaperRetriever()  # Placeholder for paper retriever if needed
        self.two_steps_agent = two_steps_agent

    def compile(self):
        """
        compile the identify workflow
        """
        self.steps.append(
            IdentifyRelevanceStep(
                llm=self.llm,
                two_steps_agent=self.two_steps_agent,
            )
        )
        self.steps.append(
            IdentifyOriginalDataStep(
                llm=self.llm,
                two_steps_agent=self.two_steps_agent,
            )
        )
        def check_original(state: IdentifyState) -> bool:
            original = state.get("original", None)
            if original is None:
                return False
            return original
        
        graph = StateGraph(IdentifyState)
        graph.add_node("identify_relevance", self.steps[0].execute)
        graph.add_node("identify_original_data", self.steps[1].execute)
        graph.add_edge(START, "identify_original_data")
        graph.add_conditional_edges("identify_original_data", check_original, {
            True: "identify_relevance", False: END
        })
        graph.add_edge("identify_relevance", END)

        self.graph = graph.compile()

    def identify(
        self, 
        pmid: str, 
        research_goal: str,
        identify_original_instructions: Optional[str] = None,
        identify_relevant_instructions: Optional[str] = None,
    ) -> bool:
        """
        Identify the relevance and original data of a paper by its PubMed ID (PMID).
        Args:
            pmid (str): The PubMed ID of the paper.
        Returns:
            IdentifyState: The state containing the results of the identification process.
        """
        title, abstract, is_preprint = self.paper_retriever.query_title_abstract_ispreprint(pmid) # query_title_abstract_ispreprint(pmid)
        if not title or not abstract or is_preprint:
            return False
        res, html_content = self.paper_retriever.query_full_text(pmid) # query_full_text(pmid)
        if not res or not html_content:
            return False
        full_text = convert_html_to_plaintext(html_content)
        if not full_text:
            return False
        

        state = IdentifyState(
            pmid=pmid,
            research_goal=research_goal,
            title=title,
            abstract=abstract,
            content=full_text,
            step_output_callback=self.step_callback,
            identify_original_instructions=identify_original_instructions or "N/A",
            identify_relevant_instructions=identify_relevant_instructions or "N/A",
        )

        s = None
        for s in self.graph.stream(
            input=state,
            stream_mode="values",
            config={"recursion_limit": 1000},
        ):
            # print(s)
            continue
        
        if s is None:
            return False
        return s.get("relevant", False) and s.get("original", False)

        
def identify_workflow(
    wf: IdentifyWorkflow,
    pmid: str, 
    research_goal: str,
    identify_original_instructions: Optional[str] = None,
    identify_relevant_instructions: Optional[str] = None,
) -> bool:
    """
    Identify the relevance and original data of a paper by its PubMed ID (PMID).
    Args:
        pmid (str): The PubMed ID of the paper.
        llm (BaseChatOpenAI): The language model to use.
        step_callback (Optional[Callable]): Callback function for step output.
        research_goal (ResearchGoalEnum): The research goal to use.
        identify_original_instructions (Optional[str]): Additional instructions for identification.
    Returns:
        bool: True if the paper is relevant and has original data, False otherwise.
    """
    return wf.identify(
        pmid=pmid,
        research_goal=research_goal,
        identify_original_instructions=identify_original_instructions,
        identify_relevant_instructions=identify_relevant_instructions,
    )