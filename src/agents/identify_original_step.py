
from typing import Callable, Optional, TypedDict
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .common_step import sskindCommonStep
from .common_agent import CommonAgent
from .common_agent_2step import CommonAgentTwoSteps, CommonAgentTwoChainSteps
from .agent_utils import IdentifyState, RESEARCH_GOAL_DICT


IDENTIFY_ORIGINAL_DATA_SYSTEM_PROMPT = ChatPromptTemplate.from_template("""
---

You are an expert in **biomedical research** and a skilled **data scientist**.

We are collecting data from published literature related to **{research_goal}**.
You will be provided with the **title** and **full text** of a scientific paper.

Your task is to determine:

1. Whether **the data** presented in the paper is **original** (i.e., newly generated and not previously published).
2. Whether the data is **publicly accessible** (i.e., available through a repository, supplementary materials, or other open sources).

---

### **Instructions**
1. If the paper is preprint version and there is not public link provided, you should assume the data is not publicly accessible.
2. If In **Data Availability** section, the data is described as **available upon request** or uses similar language, you should assume the data is not publicly accessible.
3. You should evaluate if **the data** is original and accessible, not the entire paper.

### **Important Instructions**
{important_instructions}

### **Input**
**Title:**
{title}
**Full Text:**
{full_text}
---

### **Output**
Please answer the following question:
**Is the data in this paper original and not previously published, and publicly accessible?**

Respond in the following format:

**FinalAnswer**: [Yes / No]
""")

class IdentifyOriginalDataResult(BaseModel):
    reasoning_process: Optional[str] = Field(
        description="The reasoning process used to determine relevance."
    )
    original_and_accessible: bool = Field(
        description="Indicates whether the data in the paper is original and not previously published, and publicly accessible."
    )

class IdentifyOriginalDataStep(sskindCommonStep):
    """
    This class implements the step to identify if the data in a paper is original.
    """
    def __init__(
        self, 
        llm: BaseChatOpenAI,
        two_steps_agent: bool = False,
    ):
        super().__init__(llm)
        self.llm = llm
        self.step_name = "Identify Original Data Step"
        self.two_steps_agent = two_steps_agent

    def _execute_directly(self, state: dict) -> tuple[dict | None, dict[str, int] | None]:
        typed_state: IdentifyState = IdentifyState(**state)
        pmid = typed_state.get("pmid", "N/A")
        self._print_step(typed_state, step_output=f"PMID: {pmid}")
        
        research_goal = typed_state.get("research_goal")
        important_instructions = typed_state.get("identify_original_instructions", "N/A")
        title = typed_state.get("title")
        full_text = typed_state.get("content")

        agent = CommonAgent(llm=self.llm) if not self.two_steps_agent else CommonAgentTwoChainSteps(self.llm)
        system_prompt = IDENTIFY_ORIGINAL_DATA_SYSTEM_PROMPT.format(
            research_goal=research_goal,
            title=title, 
            full_text=full_text,
            important_instructions=important_instructions,
        )
        res, _, token_usage, reasoning_process = agent.go(
            system_prompt=system_prompt,
            instruction_prompt="Before jumping to final answer, you need to explain **the reasoning process** first.\nNow, let's identify if the data is original and accessible.",
            schema=IdentifyOriginalDataResult,
        )
        typed_state["original"] = res.original_and_accessible
        self._print_step(
            typed_state,
            step_output=res.reasoning_process if reasoning_process is None else reasoning_process,
        )

        return dict(typed_state), token_usage

