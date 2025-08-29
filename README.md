# ssKIND-collection-agents

## Usage
We use [Poetry](https://python-poetry.org) for dependency management. Please make sure that you have installed Poetry and set up the environment correctly before starting development.

### setup environment
- Install dependencies from the lock file: `poetry install`

- Use the environment: You can either run commands directly with `poetry run
<command>` or open a shell with `eval $(poetry env activate)` and then run commands directly.

### prepare environment variables
- copy `.env.template` and rename to `.env`
- in `.env`, set api key and model for the desired LLM (OpenAI, Gemini or Claude), such as
```
OPENAI_4O_API_KEY=xxx
OPENAI_4O_DEPLOYMENT_NAME=xxx
AZURE_OPENAI_4O_ENDPOINT=https://...
...
```

### run tests in system_tests
1. Comment pytest skip. As the tests in system_tests need to consume API tokens, we skipped all the tests.
Therefore, if we need to run a specific test, we need to comment pytest skip fixture: `@pytest.mark.skip()`
2. Run pytest on the specific test file, e.g.
```
python -m pytest system_tests/test_identify_original_step.py
```

### run app script
1. Curate literature with `app_script.py` on Alzheimer Single Cell (Alzheimer_SingleCell)
```
python app_script.py -s Alzheimer_SingleCell
```

2. Check app_script usage
```
python app_script.py --help
```

