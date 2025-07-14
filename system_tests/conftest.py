
import logging
from typing import Optional
import pytest
import os

from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from dotenv import load_dotenv

from src.agents.agent_utils import increase_token_usage
from src.agents.constants import DEFAULT_TOKEN_USAGE
from src.config_utils import read_config_identify_original_instructions, read_config_identify_relevant_instructions, read_config_query

load_dotenv()

def get_openai():
    return ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY", None),
        model=os.getenv("OPENAI_MODEL_NAME", None),
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

@pytest.fixture(scope="session", autouse=True)
def prepare_logging():
    level = logging.INFO
    logging.basicConfig(level=level)
    file_handler = logging.FileHandler("./logs/test.log")
    file_handler.setLevel(level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

@pytest.fixture(scope="module")
def llm():
    return get_azure_openai()

@pytest.fixture(scope="module")
def sc_alzheimers_query():
    return read_config_query("Alzheimer_SingleCell")

@pytest.fixture(scope="module")
def sc_alzheimers_identify_original_instructions():
    return read_config_identify_original_instructions("Alzheimer_SingleCell")

@pytest.fixture(scope="module")
def sc_alzheimers_identify_relevant_instructions():
    return read_config_identify_relevant_instructions("Alzheimer_SingleCell")

@pytest.fixture(scope="module")
def step_callback():
    total_tokens = {**DEFAULT_TOKEN_USAGE}

    def print_step(
        step_name: Optional[str] = None,
        step_output: Optional[str] = None,
        token_usage: Optional[dict] = None,
    ):
        nonlocal total_tokens
        logger = logging.getLogger(__name__)
        if step_name is not None:
            logger.info("=" * 64)
            logger.info(step_name)
        if token_usage is not None:
            logger.info(
                f"step total tokens: {token_usage['total_tokens']}, step prompt tokens: {token_usage['prompt_tokens']}, step completion tokens: {token_usage['completion_tokens']}"
            )
            total_tokens = increase_token_usage(total_tokens, token_usage)
            logger.info(
                f"overall total tokens: {total_tokens['total_tokens']}, overall prompt tokens: {total_tokens['prompt_tokens']}, overall completion tokens: {total_tokens['completion_tokens']}"
            )
        if step_output is not None:
            logger.info(step_output)

    return print_step