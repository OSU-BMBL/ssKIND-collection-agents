# Pipeline 1 — Paper Screening

## Overview

Pipeline 1 screens PubMed papers and decides which ones are relevant to a configured research scope (e.g., Alzheimer's single-cell RNA-seq). It is implemented as a LangGraph workflow with two LLM-powered steps executed per PMID.

A paper is accepted only when **both** steps return `True`.

## Flow

```
PubMed query (eUtils)
        │
        ▼  for each PMID
   fetch title + abstract
        │
        ├── [SQLite cache hit] ──────────────────────────────┐
        │                                                     │
        ▼                                                     │
  fetch full-text HTML (NetToolkit)                          │
        │                                                     │
        ▼                                                     ▼
  IdentifyOriginalDataStep (LLM)
        │
        ├─ False ──> REJECTED (result="no")
        │
        ▼
  IdentifyRelevanceStep (LLM)
        │
        ├─ False ──> REJECTED (result="no")
        │
        ▼
  ACCEPTED (result="yes")
        │
        ▼
  append to results.txt
```

## Entry Points

### Single scope

```bash
python app_script.py -s Alzheimer_SingleCell
```

### All configured scopes

```bash
python run_all_scopes.py
```

### Output

Each run appends lines to `results.txt`:
```
<PMID>: yes   # accepted
<PMID>: no    # rejected
```

## Configuration (`config/scope_config.yaml`)

Each scope entry defines:

```yaml
Alzheimer_SingleCell:
  query: "Alzheimer's disease[Title/Abstract] AND single cell[Title/Abstract]"
  mindate: "2020/01/01"
  maxdate: "2026/12/31"
  identify_original_instructions: |
    Accept papers that deposited original scRNA-seq or spatial transcriptomics
    data in a public repository (GEO, ArrayExpress, Zenodo, etc.)...
  identify_relevant_instructions: |
    Accept papers studying Alzheimer's disease with single-cell resolution...
```

Add a new scope by appending a new YAML entry and running `app_script.py -s <NewScope>`.

## Step Details

### Step 1 — IdentifyOriginalDataStep

**File:** `src/agents/identify_original_step.py`

Checks whether the paper describes original data (not a meta-analysis or review) that is publicly deposited. Uses the paper's abstract and, if available, full text.

Key signals it looks for:
- GEO / ArrayExpress / Zenodo / figshare accession numbers
- Language confirming data availability ("data are available at", "deposited in")
- Absence of re-analysis of previously published data

**Output field:** `original: bool`

### Step 2 — IdentifyRelevanceStep

**File:** `src/agents/identify_relevant_step.py`

Checks whether the paper's single-cell data is relevant to the configured scope. Runs only if Step 1 returns `True`.

Uses `identify_relevant_instructions` from `scope_config.yaml` to define what counts as relevant for each disease area.

**Output field:** `relevant: bool`

## Key Source Files

| File | Purpose |
|---|---|
| `app_script.py` | CLI: parses args, builds LLM, calls `identify_workflow()` |
| `run_all_scopes.py` | Spawns `app_script.py` as subprocess for each scope |
| `src/workflow/identify_workflow.py` | `IdentifyWorkflow` LangGraph definition |
| `src/agents/identify_original_step.py` | Step 1 LLM agent |
| `src/agents/identify_relevant_step.py` | Step 2 LLM agent |
| `src/agents/common_agent.py` | Base single-call LLM agent (5× retry) |
| `src/agents/common_agent_2step.py` | Two-call CoT + structured-output agent |
| `src/agents/agent_utils.py` | `IdentifyState` TypedDict, token tracking |
| `src/paper_query/pubmed_query.py` | PubMed NCBI eUtils queries, SQLite cache |
| `src/paper_query/article_retriever.py` | Full-text HTML fetcher via NetToolkit |
| `src/database/pmid_paper_db.py` | SQLite: title / abstract / HTML per PMID |
| `config/scope_config.yaml` | All scope definitions |

## LLM Agent Pattern

All Pipeline 1 agents follow this pattern:

```python
agent = CommonAgent(llm=llm)
result, input_tokens, output_tokens, latency = agent.go(
    system_prompt=system_prompt,
    instruction_prompt=instruction_prompt,
    schema=MyPydanticSchema,
)
```

`MyPydanticSchema` must have a `reasoning_process: str` field (for CoT) plus a boolean result field. The agent retries up to 5 times on LLM errors using `tenacity`.

### Two-step variant

`CommonAgentTwoChainSteps` makes two sequential LLM calls: the first produces free-form reasoning; the second calls a dedicated extraction prompt that cites the first call's output. Activated by `two_steps_agent=True` in `IdentifyWorkflow`.

## State Object (`IdentifyState`)

Passed through the LangGraph graph nodes:

| Field | Type | Set by |
|---|---|---|
| `pmid` | str | caller |
| `title` | str | caller |
| `abstract` | str | caller |
| `content` | str | caller (full-text plain) |
| `research_goal` | str | caller |
| `identify_original_instructions` | str | scope config |
| `identify_relevant_instructions` | str | scope config |
| `original` | bool | IdentifyOriginalDataStep |
| `relevant` | bool | IdentifyRelevanceStep |
| `step_output_callback` | callable | app_script.py |

## Caching

`PubMedPaperRetriever` caches NCBI API responses and full-text HTML in a SQLite database at `$DATA_FOLDER/database/pmid_paper.db`. Subsequent runs skip network calls for cached PMIDs.

## Tests

| Test file | What it tests |
|---|---|
| `system_tests/test_identify_original_step.py` | IdentifyOriginalDataStep (LLM, skipped by default) |
| `system_tests/test_identify_relevance_step.py` | IdentifyRelevanceStep (LLM, skipped by default) |
| `system_tests/test_identify_workflow.py` | End-to-end workflow (LLM, skipped by default) |
| `system_tests/test_pubmed_query.py` | PubMed queries and HTML extraction |
| `system_tests/test_pubmed_fulltext.py` | Full-text retrieval |
| `system_tests/test_article_retriever.py` | NetToolkit article retriever |
| `system_tests/test_read_config.py` | scope_config.yaml parsing |
