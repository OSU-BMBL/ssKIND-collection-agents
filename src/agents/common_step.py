
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, TypedDict
import logging
from langchain_openai.chat_models.base import BaseChatOpenAI

from .constants import DEFAULT_TOKEN_USAGE

logger = logging.getLogger(__name__)

class CommonState(TypedDict):
    """
    CommonState is a TypedDict that defines the structure of the state
    used in the CommonStep class.
    """
    llm: Optional[BaseChatOpenAI]
    step_output_callback: Optional[Callable]

class CommonStep(ABC):
    """
    CommonStep is a base class for defining common steps in a workflow.
    It provides methods to execute the step and handle exceptions.
    """

    def __init__(self):
        super().__init__()
        self.step_name = ""

    def enter_step(self, state):
        if state["step_output_callback"] is None:
            return
        state["step_output_callback"](
            step_name=self.step_name, 
        )

    def leave_step(self, state, token_usage: Optional[dict[str, int]] = None):
        if state["step_output_callback"] is None:
            return
        if token_usage is not None:
            state["step_output_callback"](token_usage=token_usage)

    def execute(self, state):
        """
        Execute the step. This method should be overridden by subclasses.
        """
        self.enter_step(state)
        state, token_usage = self._execute_directly(state)
        self.leave_step(state, token_usage)
        return state

    def _print_step(
        self,
        state,
        step_name: str | None = None,
        step_output: str | None = None,
        token_usage: dict | object | None = None,
    ):
        step_callback = state["step_output_callback"]
        if step_callback is None:
            return
        # convert token_usage to dict
        if token_usage is not None and not isinstance(token_usage, dict):
            token_usage = vars(token_usage)
            # In case token_usage.total_tokens is 0
            token_usage = { **DEFAULT_TOKEN_USAGE, **token_usage }
        step_callback(
            step_name=step_name,
            step_output=step_output,
            token_usage=token_usage,
        )                

    @abstractmethod
    def _execute_directly(self, state: dict) -> tuple[dict | None, dict[str, int] | None]:
        """
        Execute the step directly. This method should be overridden by subclasses.
        Args:
            state (CommonState): The state of the workflow.
        Returns:
            tuple[dict, dict[str, int]]: The updated state and token usage.
        """
        pass


class sskindCommonStep(CommonStep):
    """
    This class is a placeholder for common step functionality in the sskind agent.
    It is currently empty and can be extended in the future.
    """
    def __init__(self, llm: BaseChatOpenAI):
        super().__init__()
        self.llm = llm



