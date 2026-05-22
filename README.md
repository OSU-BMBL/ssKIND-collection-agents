# ssKIND-collection-agents

Automated biomedical literature curation and single-cell data processing system for neurological disease research (Alzheimer's, Parkinson's, MS, ALS, etc.).

**Pipeline 1** screens PubMed papers and identifies those with original, publicly accessible single-cell RNA-seq or spatial transcriptomics data.

**Pipeline 2** processes each accepted dataset end-to-end: metadata extraction → data download → format conversion → QC → doublet removal → cell-type annotation → atlas assembly.

## Usage

We use [Poetry](https://python-poetry.org) for dependency management. Please make sure that you have installed Poetry and set up the environment correctly before starting development.

> **Python version:** Python 3.13 is recommended. Use a conda environment such as `sskind-agent-py313` (Python 3.13) to ensure all dependencies install correctly.

### Setup environment

Install dependencies from the lock file:
```bash
poetry install
```

Use the environment: run commands directly with `poetry run <command>` or open a shell with:
```bash
eval $(poetry env activate)
```

### Prepare environment variables

Copy `.env.template` and rename to `.env`:
```bash
cp .env.template .env
```

Then fill in the required variables:

#### Pipeline 1 — Paper Screening

| Variable | Purpose |
|---|---|
| `OPENAI_4O_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_4O_ENDPOINT` | Azure endpoint URL |
| `OPENAI_4O_API_VERSION` | API version string |
| `OPENAI_4O_DEPLOYMENT_NAME` | Azure deployment name |
| `OPENAI_4O_MODEL` | Model name (e.g. `gpt-4o`) |
| `OPENAI_MAX_OUTPUT_TOKENS` | Max output tokens (default `16380`) |
| `BASE_URL` | NetToolkit server URL (default `http://127.0.0.1:3001`) |

#### Pipeline 2 — Data Processing (additional variables)

| Variable | Purpose |
|---|---|
| `DATA_FOLDER` | Root folder for all pipeline data (default: project root) |
| `HUMAN_GENE_MART_PATH` | Path to human gene-ID→symbol TSV (columns: `gene_ids`, `gene_symbols`) |
| `MOUSE_GENE_MART_PATH` | Path to mouse gene-ID→symbol TSV |
| `HUMAN_MAPMYCELLS_TAXONOMY_PATH` | Path to human MapMyCells precomputed stats file (required for cell-type annotation) |
| `MOUSE_MAPMYCELLS_TAXONOMY_PATH` | Path to mouse MapMyCells precomputed stats file (required for cell-type annotation) |

If `HUMAN_MAPMYCELLS_TAXONOMY_PATH` / `MOUSE_MAPMYCELLS_TAXONOMY_PATH` are not set, cell-type annotation is skipped and all cells are labelled `"Unknown"` — downstream atlas steps still run normally.

### NetToolkit (required for full-text retrieval)

`app_script.py` and `app_processing.py` rely on [NetToolkit](https://hub.docker.com/r/frankfeng78/nettoolkit/tags) to download PubMed literature. Start it before running either pipeline:

```bash
docker pull frankfeng78/nettoolkit:0.1.7
docker run -d --name nettoolkit -p3001:3001 frankfeng78/nettoolkit:0.1.7
```

### Run tests

Most tests in `system_tests/` run without API tokens and do not require LLM access:

```bash
eval $(poetry env activate)
python -m pytest system_tests/
```

To run a specific LLM-backed test, first remove or comment out its `@pytest.mark.skip()` decorator, then:

```bash
python -m pytest system_tests/test_metadata_extractor_step.py
```

---

## Pipeline 1 — Paper Screening

Screens PubMed for papers matching a configured scope and writes accepted PMIDs to `results.txt`.

```bash
# Screen a single scope
python app_script.py -s Alzheimer_SingleCell

# List available scopes
python app_script.py --help

# Screen all configured scopes
python run_all_scopes.py
```

Scopes are defined in `config/scope_config.yaml`. Each scope specifies a PubMed query, date range, and LLM instructions for originality and relevance checks.

---

## Pipeline 2 — Data Processing

Processes one or more accepted PMIDs through the full pipeline and merges results into an atlas.

```bash
# Process one or more PMIDs and merge into a named atlas
python app_processing.py -p 39578645 39607927 --atlas-name alzheimer

# Process without the final atlas merge step
python app_processing.py -p 39578645 --skip-atlas

# Override the data folder (default: DATA_FOLDER env var or project root)
python app_processing.py -p 39578645 -o /path/to/data
```

Steps are **idempotent** — re-running the same PMID skips steps whose output files already exist. If full text cannot be fetched but `0.metadata/{pmid}.json` already exists on disk, the paper is still processed from the cached metadata.

### Output directory layout

```
$DATA_FOLDER/
  0.metadata/        # {pmid}.json — extracted dataset list per paper
  1.manifest/        # {dataset_id}.json — file manifest per dataset
  2.raw/             # {dataset_id}/ — downloaded files + conversion_config.json
  3.h5ad/            # {dataset_id}.h5ad — converted human h5ad
  3.Mh5ad/           # {dataset_id}.h5ad — converted mouse h5ad
  4.qc/              # QC-annotated and filtered h5ad + QC reports
  5.doublet/         # Doublet-removed h5ad + cell-type annotation labels
  6.labeled/         # h5ad with cell_type column merged in
  7.atlas_clean/     # Cleaned h5ad ready for atlas inclusion
  8.atlas/           # Final merged atlas h5ad
```

### Raw Cell Ranger data

When the downloaded data is an unfiltered Cell Ranger output (raw barcode-gene matrix), the pipeline automatically applies a **500-UMI empty-droplet filter** before QC. This removes the millions of empty barcodes that Cell Ranger includes in its raw output, keeping only real cells. The filter is applied when the conversion config reports `normalization_status = "raw_counts"` and `data_type` is one of `h5`, `10x`, or `10x_matrix`.

### Cell-type annotation

Cell-type annotation uses [MapMyCells](https://github.com/AllenInstitute/cell_type_mapper) with Allen Institute taxonomies. Configure the paths via `HUMAN_MAPMYCELLS_TAXONOMY_PATH` and `MOUSE_MAPMYCELLS_TAXONOMY_PATH`. If these are not set, annotation is skipped gracefully and cells receive `cell_type = "Unknown"`. The atlas cleaning step respects this and does not remove cells when annotation was not run.
