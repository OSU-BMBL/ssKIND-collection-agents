# ssKIND-collection-agents

## Project Purpose

Automated biomedical literature curation and single-cell data processing system for neurological disease research (Alzheimer's, Parkinson's, MS, ALS, etc.).

**Pipeline 1** screens PubMed papers and filters those with original, publicly accessible single-cell RNA-seq or spatial transcriptomics data.

**Pipeline 2** processes each accepted dataset end-to-end: metadata extraction → data download → format conversion → QC → doublet removal → cell-type annotation → atlas assembly.

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
| `DATA_FOLDER` | Root folder for all pipeline data (default: project root) |
| `HUMAN_GENE_MART_PATH` | Path to human gene-ID→symbol TSV (columns: gene_ids, gene_symbols) |
| `MOUSE_GENE_MART_PATH` | Path to mouse gene-ID→symbol TSV |
| `HUMAN_MAPMYCELLS_TAXONOMY_PATH` | Path to human MapMyCells precomputed stats file |
| `MOUSE_MAPMYCELLS_TAXONOMY_PATH` | Path to mouse MapMyCells precomputed stats file |

### NetToolkit (required for full-text retrieval)

```bash
docker pull frankfeng78/nettoolkit:0.1.7
docker run -d --name nettoolkit -p3001:3001 frankfeng78/nettoolkit:0.1.7
```

## Running Pipeline 1 (Paper Screening)

```bash
# Run a single scope
python app_script.py -s Alzheimer_SingleCell

# List available scopes
python app_script.py --help

# Run multiple scopes
python run_all_scopes.py
```

Output is appended to `results.txt` — one line per PMID indicating relevance.

## Running Tests

System tests that call real APIs or LLMs are skipped by default.

```bash
# Run all non-LLM tests
eval $(poetry env activate)
python -m pytest system_tests/

# Enable an LLM test
# 1. Remove or comment out @pytest.mark.skip() in the test file
# 2. Run the specific test file
python -m pytest system_tests/test_metadata_extractor_step.py
```

## Architecture

### Pipeline 1 — Paper Screening

LangGraph two-step workflow per PMID (`src/workflow/identify_workflow.py`):

```
START
  └─> IdentifyOriginalDataStep   # Is data original & publicly accessible?
        ├─ No  ──> END (rejected)
        └─ Yes ──> IdentifyRelevanceStep  # Is it relevant to the scope?
                        └─> END
```

A paper is accepted only if **both** steps return `True`.

**Key components:**

| Path | Role |
|---|---|
| `app_script.py` | CLI entry point; wires LLM, config, and workflow |
| `run_all_scopes.py` | Batch runner; invokes `app_script.py` per scope |
| `config/scope_config.yaml` | Per-scope PubMed query, date range, LLM instructions |
| `src/config_utils.py` | Reads `scope_config.yaml` |
| `src/workflow/identify_workflow.py` | `IdentifyWorkflow` + `identify_workflow()` helper |
| `src/agents/identify_original_step.py` | Step 1: originality & accessibility check |
| `src/agents/identify_relevant_step.py` | Step 2: relevance check |
| `src/agents/common_agent.py` | Single-call LLM agent with structured output |
| `src/agents/common_agent_2step.py` | Two-call agent: CoT then structured extraction |
| `src/agents/common_step.py` | Abstract base class for workflow steps |
| `src/agents/agent_utils.py` | `IdentifyState` TypedDict, token usage helpers |
| `src/paper_query/pubmed_query.py` | PubMed NCBI eUtils queries + SQLite cache |
| `src/paper_query/article_retriever.py` | Full-text HTML download via NetToolkit |
| `src/database/pmid_paper_db.py` | SQLite cache for title/abstract/HTML per PMID |

### Pipeline 2 — Data Processing

File-based pipeline where each step reads its inputs and writes its outputs to disk under `$DATA_FOLDER/`. Steps are idempotent — they skip if their output files already exist.

**Directory layout:**

```
$DATA_FOLDER/
  0.metadata/          # {pmid}.json — extracted dataset list per paper
  1.manifest/          # {dataset_id}.json — file manifest per dataset
  2.raw/               # {dataset_id}/ — downloaded files + conversion_config.json
  3.h5ad/              # {dataset_id}.h5ad — converted human h5ad
  3.Mh5ad/             # {dataset_id}.h5ad — converted mouse h5ad
  4.qc/                # {dataset_id}_annotated.h5ad, _qc_report.json,
                       #   _thresholds.json, {dataset_id}.h5ad (filtered)
  5.doublet/           # {dataset_id}.h5ad, _doublet_result.json,
                       #   _annotation_config.json, _labels.csv
  6.labeled/           # {dataset_id}.h5ad — with cell_type column
  7.atlas_clean/       # {dataset_id}.h5ad — cleaned, ready for atlas
  8.atlas/             # {atlas_name}.h5ad — final merged atlas
```

**Step sequence:**

| Step | Class | Location | LLM | Input → Output |
|---|---|---|---|---|
| A0 | `MetadataExtractorStep` | `src/agents/` | Yes | paper full-text → `0.metadata/{pmid}.json` |
| A1 | `RepositoryAnalystStep` | `src/agents/` | Yes | metadata → `1.manifest/{dataset_id}.json` |
| A2 | `DataDownloaderStep` | `src/data_processing/` | No | manifest → `2.raw/{dataset_id}/` |
| B1 | `FormatAnalyzerStep` | `src/agents/` | Yes | raw files → `2.raw/{id}/conversion_config.json` |
| B2 | `FormatConverterStep` | `src/data_processing/` | No | raw files → `3.h5ad/` or `3.Mh5ad/` |
| C1 | `CountQCStep` | `src/data_processing/` | No | h5ad → `4.qc/_annotated.h5ad` + `_qc_report.json` |
| C2 | `QCReviewerStep` | `src/agents/` | Yes | QC report → `4.qc/_thresholds.json` |
| C3 | `QCFilterStep` | `src/data_processing/` | No | annotated h5ad + thresholds → `4.qc/{id}.h5ad` |
| D | `DoubletDetectionStep` | `src/data_processing/` | No | QC h5ad → `5.doublet/{id}.h5ad` |
| E1 | `AnnotationConfigStep` | `src/agents/` | Yes | conversion config → `5.doublet/_annotation_config.json` |
| E2 | `CellTypeAnnotationStep` | `src/data_processing/` | No | doublet h5ad → `5.doublet/_labels.csv` |
| F | `LabelMergerStep` | `src/data_processing/` | No | doublet h5ad + labels → `6.labeled/{id}.h5ad` |
| G | `AtlasCleanerStep` | `src/data_processing/` | No | labeled h5ad → `7.atlas_clean/{id}.h5ad` |
| H | `AtlasMergerStep` | `src/data_processing/` | No | all cleaned h5ad → `8.atlas/{name}.h5ad` |

### Agent Variants

- `CommonAgent` — single LLM call with `with_structured_output(schema)`; retries up to 5× via `tenacity`
- `CommonAgentTwoSteps` — two LLM calls: free-form CoT first, then structured extraction
- `CommonAgentTwoChainSteps` — variant where the extraction step uses a dedicated prompt

`two_steps_agent=True` in `IdentifyWorkflow` selects `CommonAgentTwoChainSteps`.

### LLM Prompt Conventions

- System prompts use `ChatPromptTemplate.from_template()`
- Curly braces `{}` in non-template strings must be escaped before passing to LangChain
- All structured output schemas are Pydantic `BaseModel` subclasses with a `reasoning_process` field
- LLM steps are idempotent: they load from disk if their output JSON already exists

### Caching

- **Pipeline 1**: `PubMedPaperRetriever` caches title/abstract/HTML in SQLite at `$DATA_FOLDER/database/pmid_paper.db`
- **Pipeline 2**: every step writes a JSON result file alongside its h5ad output; presence of both files triggers the skip-if-done guard

## Adding a New Scope (Pipeline 1)

1. Add a new entry to `config/scope_config.yaml` with `query`, `mindate`, `maxdate`, `identify_original_instructions`, `identify_relevant_instructions`
2. Optionally add the name to `scopes_to_run` in `run_all_scopes.py`
3. Run: `python app_script.py -s <NewScopeName>`
