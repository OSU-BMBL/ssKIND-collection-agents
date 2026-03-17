# ssKIND-collection-agents

## Project Purpose

Automated biomedical literature curation pipeline that identifies PubMed papers relevant to neurological disease research (Alzheimer's, Parkinson's, MS, ALS, etc.). For each scope, it filters papers that have **original, publicly accessible** single-cell RNA-seq or spatial transcriptomics data.

## Environment Setup

```bash
# Install dependencies
poetry install

# Activate environment
eval $(poetry env activate)

# Copy and fill in credentials
cp .env.template .env
```

### Required env vars (`.env`)
| Variable | Purpose |
|---|---|
| `OPENAI_4O_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_4O_ENDPOINT` | Azure endpoint URL |
| `OPENAI_4O_API_VERSION` | API version string |
| `OPENAI_4O_DEPLOYMENT_NAME` | Azure deployment name |
| `OPENAI_4O_MODEL` | Model name (e.g. `gpt-4o`) |
| `OPENAI_MAX_OUTPUT_TOKENS` | Max output tokens (default `16380`) |
| `BASE_URL` | NetToolkit server URL (default `http://127.0.0.1:3001`) |
| `DATA_FOLDER` | Folder for SQLite cache DB (default: project root) |

### NetToolkit (required for full-text retrieval)
Full-text download requires a local Docker service:
```bash
docker pull frankfeng78/nettoolkit:0.1.7
docker run -d --name nettoolkit -p3001:3001 frankfeng78/nettoolkit:0.1.7
```

## Running the Pipeline

```bash
# Run a single scope
python app_script.py -s Alzheimer_SingleCell

# List available scopes
python app_script.py --help

# Run multiple scopes (edit scopes_to_run list first)
python run_all_scopes.py
```

Output is appended to `results.txt` — one line per PMID indicating relevance.

## Running Tests

System tests consume real API tokens and are skipped by default. To run one:
1. Comment out `@pytest.mark.skip()` in the test file
2. Run: `python -m pytest system_tests/test_identify_original_step.py`

## Architecture

### Pipeline Flow (`src/workflow/identify_workflow.py`)

LangGraph two-step workflow executed per PMID:

```
START
  └─> IdentifyOriginalDataStep   # Is data original & publicly accessible?
        ├─ No  ──> END (rejected)
        └─ Yes ──> IdentifyRelevanceStep  # Is it relevant to the research scope?
                        └─> END
```

A paper is accepted only if **both** steps return `True`.

### Key Components

| Path | Role |
|---|---|
| `app_script.py` | CLI entry point; wires LLM, config, and workflow |
| `run_all_scopes.py` | Batch runner; invokes `app_script.py` per scope via subprocess |
| `config/scope_config.yaml` | Per-scope PubMed query, date range, and LLM instructions |
| `src/config_utils.py` | Reads `scope_config.yaml` |
| `src/workflow/identify_workflow.py` | `IdentifyWorkflow` class + `identify_workflow()` helper |
| `src/agents/identify_original_step.py` | Step 1: originality & accessibility check |
| `src/agents/identify_relevant_step.py` | Step 2: relevance check |
| `src/agents/common_agent.py` | Single-call LLM agent with structured output |
| `src/agents/common_agent_2step.py` | Two-call agent: CoT reasoning then structured extraction |
| `src/agents/common_step.py` | Abstract base class for workflow steps |
| `src/agents/agent_utils.py` | `IdentifyState` TypedDict, token usage helpers |
| `src/paper_query/pubmed_query.py` | PubMed NCBI eUtils queries; `PubMedPaperRetriever` (with DB cache) |
| `src/paper_query/article_retriever.py` | Full-text HTML download via NetToolkit |
| `src/database/pmid_paper_db.py` | SQLite cache for title/abstract/HTML per PMID |
| `src/log_utils.py` | Logger initialization |

### Agent Variants

- `CommonAgent` — single LLM call, `with_structured_output(schema)`; retries up to 5× with `tenacity`
- `CommonAgentTwoSteps` — two LLM calls: free-form CoT first, then structured output using that reasoning
- `CommonAgentTwoChainSteps` — variant of above where the final step uses a dedicated extraction prompt (rather than the original system prompt)

`two_steps_agent=True` in `IdentifyWorkflow` selects `CommonAgentTwoChainSteps` for both steps.

### State (`IdentifyState`)

TypedDict passed through the LangGraph graph:
- `pmid`, `title`, `abstract`, `content` (full text plain), `research_goal`
- `identify_original_instructions`, `identify_relevant_instructions` (from config)
- `original: bool`, `relevant: bool` (set by each step)
- `step_output_callback` (optional; used by `app_script.py` for logging)

### Caching

`PubMedPaperRetriever` wraps raw query functions and caches results in a SQLite DB at `$DATA_FOLDER/database/pmid_paper.db`. This avoids redundant API/network calls when re-running.

## Adding a New Scope

1. Add a new entry to `config/scope_config.yaml` with:
   - `query`: PubMed search string
   - `mindate` / `maxdate`: date range (`"YYYY/MM/DD"`)
   - `identify_original_instructions`: criteria for Step 1
   - `identify_relevant_instructions`: criteria for Step 2
2. Optionally add the scope name to `scopes_to_run` in `run_all_scopes.py`
3. Run: `python app_script.py -s <NewScopeName>`

## LLM Prompt Conventions

- System prompts are built with `ChatPromptTemplate.from_template()`
- Curly braces `{}` in non-template strings must be escaped as `()`  before passing to LangChain (see `system_prompt.replace("{", "(")` in agent code)
- Structured output schemas are Pydantic `BaseModel` subclasses with a `reasoning_process` field and a boolean result field
