# Pipeline 2 — Data Processing

## Overview

Pipeline 2 takes the accepted PMIDs from Pipeline 1 and processes their single-cell datasets end-to-end — from metadata extraction and file download all the way through QC filtering, doublet removal, cell-type annotation, and atlas assembly.

All steps communicate through **intermediate files on disk** under `$DATA_FOLDER/`. Every step is **idempotent**: it checks whether its output already exists and skips gracefully if so.

Steps marked **LLM** make one LangChain call per dataset; all others are purely programmatic.

## Directory Layout

```
$DATA_FOLDER/
  0.metadata/
    {pmid}.json                        # list of datasets extracted from paper
  1.manifest/
    {dataset_id}.json                  # file manifest (URLs, sizes, format)
  2.raw/
    {dataset_id}/
      {downloaded files}
      conversion_config.json           # LLM-chosen format & species info
      conversion_result.json
  3.h5ad/
    {dataset_id}.h5ad                  # converted Human h5ad
  3.Mh5ad/
    {dataset_id}.h5ad                  # converted Mouse h5ad
  4.qc/
    {dataset_id}_annotated.h5ad        # h5ad with QC metrics in .obs
    {dataset_id}_qc_report.json        # summary stats + suggested thresholds
    {dataset_id}_thresholds.json       # LLM-approved filter thresholds
    {dataset_id}.h5ad                  # QC-filtered h5ad
    {dataset_id}_filter_result.json
  5.doublet/
    {dataset_id}.h5ad                  # doublet-filtered h5ad
    {dataset_id}_doublet_result.json
    {dataset_id}_annotation_config.json  # LLM annotation decision
    {dataset_id}_labels.csv            # MapMyCells cell-type assignments
    {dataset_id}_annotation_result.json
  6.labeled/
    {dataset_id}.h5ad                  # h5ad with cell_type column
    {dataset_id}_merge_result.json
  7.atlas_clean/
    {dataset_id}.h5ad                  # cleaned, Unknown-removed, ready for atlas
    {dataset_id}_clean_result.json
  8.atlas/
    {atlas_name}.h5ad                  # final merged + normalised atlas
    {atlas_name}_merge_result.json
```

## Step-by-Step Reference

### A0 — MetadataExtractorStep `[LLM]`

**File:** `src/agents/metadata_extractor_step.py`
**Input:** PMID, paper title, full-text
**Output:** `0.metadata/{pmid}.json`

Extracts the list of datasets described in the paper. For each dataset it records species, technology, data format hint, repository accession IDs (GEO, etc.), tissue type, and atlas eligibility.

Each dataset is assigned a `dataset_id` of the form `{pmid}_{nn}` (e.g., `39578645_01`).

**Schema: `ExtractedDataset`**

| Field | Type | Meaning |
|---|---|---|
| `species` | str | "Human", "Mouse", "Other" |
| `technology` | str | "10x Chromium", "Visium", etc. |
| `data_format_hint` | str | file format hinted in the paper |
| `accession_ids` | list[str] | GEO/ArrayExpress IDs |
| `repository` | str | "GEO", "ArrayExpress", etc. |
| `normalization_hint` | str | "raw_counts", "normalized", "unknown" |
| `n_samples` | int? | number of samples if reported |
| `n_cells_reported` | int? | cell count if reported |
| `atlas_eligible` | bool | True if suitable for atlas |
| `tissue_type` | str | tissue or brain region |

---

### A1 — RepositoryAnalystStep `[LLM]`

**File:** `src/agents/repository_analyst_step.py`
**Input:** `0.metadata/{pmid}.json` (one dataset entry)
**Output:** `1.manifest/{dataset_id}.json`

Fetches the GEO supplemental file listing for the dataset's accession ID and asks the LLM to identify which files should be downloaded, confirm format, and flag if raw data is unavailable.

GEO URL formula: `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE{prefix}nnn/{accession}/suppl/`
where `prefix` = all digits except the last three.

Falls back to NCBI ESearch by PMID if no accession is available.

**Output JSON fields:** `files` (list of `ManifestFile`), `confirmed_format`, `raw_data_available`, `download_notes`.

---

### A2 — DataDownloaderStep

**File:** `src/data_processing/data_downloader.py`
**Input:** `1.manifest/{dataset_id}.json`
**Output:** `2.raw/{dataset_id}/{filename}` (each file), `2.raw/{dataset_id}/download_status.json`

Downloads all files listed in the manifest using streaming HTTP with a 1 MB chunk size. Uses a HEAD request to check `Content-Length` for skip-if-size-matches idempotency.

Each file in `download_status.json` has `status: "success" | "skipped" | "failed"`.

---

### B1 — FormatAnalyzerStep `[LLM]`

**File:** `src/agents/format_analyzer_step.py`
**Input:** `1.manifest/{dataset_id}.json`, `2.raw/{dataset_id}/download_status.json`
**Output:** `2.raw/{dataset_id}/conversion_config.json`

Presents the LLM with the downloaded file listing, file sizes, and a text-peek of any small file (< 10 KB, gzip-aware) to decide the exact conversion strategy.

**Output `conversion_config.json` fields:**

| Field | Values |
|---|---|
| `data_type` | `"10x"`, `"csv"`, `"tsv"`, `"csv.gz"`, `"h5"`, `"h5ad"`, `"rds"`, `"rdata"`, … |
| `primary_file` | filename of the main count matrix |
| `species` | `"Human"`, `"Mouse"`, `"Other"` |
| `gene_mapping_needed` | bool — True if Ensembl IDs need mapping |
| `normalization_status` | `"raw_counts"`, `"normalized"`, `"unknown"` |
| `requires_r_extraction` | bool — True for RDS / RData |
| `special_handling` | free-text note or null |

---

### B2 — FormatConverterStep

**File:** `src/data_processing/format_converter.py`
**Input:** `2.raw/{dataset_id}/conversion_config.json`, raw data files
**Output:** `3.h5ad/{id}.h5ad` (Human) or `3.Mh5ad/{id}.h5ad` (Mouse), `2.raw/{dataset_id}/conversion_result.json`

Wraps `SingleCellConverter` which supports formats: `10x`, `10x_matrix`, `csv`, `tsv`, `txt`, `csv.gz`, `tsv.gz`, `mtx`, `h5`, `h5ad`.

If `requires_r_extraction=True` the step writes `status: "requires_r_extraction"` and skips — these datasets need an R/rpy2 path outside this Python pipeline.

Gene-ID → symbol mapping uses TSV files configured via:
- `HUMAN_GENE_MART_PATH` (columns: `gene_ids`, `gene_symbols`)
- `MOUSE_GENE_MART_PATH`

The `looks_logged()` heuristic detects whether data has already been log-normalised (values ≤ 20 and non-integer → likely log1p).

Each output h5ad has `Dataset_id` and `Pubmed_id` columns in `.obs`.

---

### C1 — CountQCStep

**File:** `src/data_processing/count_qc_step.py`
**Input:** `3.h5ad/{id}.h5ad` or `3.Mh5ad/{id}.h5ad`
**Output:** `4.qc/{id}_annotated.h5ad`, `4.qc/{id}_qc_report.json`

Computes QC metrics using `sc.pp.calculate_qc_metrics()`:
- `total_counts` — total UMI counts per cell
- `n_genes_by_counts` — number of expressed genes per cell
- `pct_counts_mt` — mitochondrial gene percentage (Human/Mouse only)
- `pct_counts_ribo` — ribosomal gene percentage (Human/Mouse only)

Mitochondrial genes are detected by `MT-` prefix; ribosomal by `RPS` / `RPL` prefix.

The report JSON includes per-metric statistics (mean, median, p5, p95) and suggested default thresholds passed to Step C2.

**Default suggested thresholds (Human / Mouse):**

| Threshold | Value |
|---|---|
| `min_genes` | 200 |
| `min_cells` | 3 |
| `max_genes` | 10 000 |
| `min_total_counts` | 500 |
| `max_total_counts` | 100 000 |
| `max_pct_mt` | 5.0 % |
| `max_pct_ribo` | 20.0 % |

**Other species** only use `min_genes`, `min_cells`, `max_total_counts=10 000`.

---

### C2 — QCReviewerStep `[LLM]`

**File:** `src/agents/qc_reviewer_step.py`
**Input:** `4.qc/{id}_qc_report.json`
**Output:** `4.qc/{id}_thresholds.json`

Presents the QC summary statistics to an LLM that decides whether to keep or adjust the suggested thresholds for this specific dataset. The LLM also decides `approved: bool` — a dataset can be rejected outright if the data is fundamentally broken (all zeros, extreme artefacts).

Guidelines encoded in the prompt:
- Raise `max_pct_mt` for cardiac / muscle tissue (high MT is biological)
- Raise `max_total_counts` for large / complex cells
- Lower `min_total_counts` for nucleus-seq / low-input data

---

### C3 — QCFilterStep

**File:** `src/data_processing/qc_filter_step.py`
**Input:** `4.qc/{id}_annotated.h5ad`, `4.qc/{id}_thresholds.json`
**Output:** `4.qc/{id}.h5ad`, `4.qc/{id}_filter_result.json`

Applies the LLM-approved thresholds from C2. Returns `status: "rejected"` for datasets that the reviewer flagged as `approved=false`.

The result JSON records `n_cells_before`, `n_cells_after`, and `pct_cells_kept`.

---

### D — DoubletDetectionStep

**File:** `src/data_processing/doublet_detection_step.py`
**Input:** `4.qc/{id}.h5ad`
**Output:** `5.doublet/{id}.h5ad`, `5.doublet/{id}_doublet_result.json`

Runs [Scrublet](https://github.com/swolock/scrublet) with default parameters:
- `expected_doublet_rate = 0.06`
- `sim_doublet_ratio = 2.0`
- Score cutoff for filtering: `0.3`

If the dataset has fewer than 30 cells or features, Scrublet is skipped (all cells kept, marked as non-doublets).

Adds `scrublet_score`, `scrublet_call`, and `predicted_doublets` columns to `.obs`.

---

### E1 — AnnotationConfigStep `[LLM]`

**File:** `src/agents/annotation_config_step.py`
**Input:** `2.raw/{dataset_id}/conversion_config.json`
**Output:** `5.doublet/{id}_annotation_config.json`

Decides whether MapMyCells cell-type annotation should be run and, if so, which taxonomy to use:
- `human_whole_brain` — for Human scRNA-seq / snRNA-seq
- `mouse_whole_brain` — for Mouse scRNA-seq / snRNA-seq
- `none` — skip annotation (e.g., spatial-only, other species, too few cells)

---

### E2 — CellTypeAnnotationStep

**File:** `src/data_processing/cell_type_annotation_step.py`
**Input:** `5.doublet/{id}.h5ad`, `5.doublet/{id}_annotation_config.json`
**Output:** `5.doublet/{id}_labels.csv`, `5.doublet/{id}_annotation_result.json`

Runs the [Allen Institute MapMyCells](https://portal.brain-map.org/atlases-and-data/bkp/mapmycells) hierarchical mapping tool (`cell_type_mapper` package) using precomputed taxonomy reference files.

**Required env vars:**
- `HUMAN_MAPMYCELLS_TAXONOMY_PATH` — path to human precomputed stats
- `MOUSE_MAPMYCELLS_TAXONOMY_PATH` — path to mouse precomputed stats

If the package is not installed or the taxonomy files are not configured, the step returns `status: "requires_external_tool"` and the pipeline continues with all cells marked "Unknown" in Step F.

**Install:** `pip install cell-type-mapper`

---

### F — LabelMergerStep

**File:** `src/data_processing/label_merger_step.py`
**Input:** `5.doublet/{id}.h5ad`, `5.doublet/{id}_labels.csv`
**Output:** `6.labeled/{id}.h5ad`, `6.labeled/{id}_merge_result.json`

Joins the MapMyCells labels CSV onto `.obs` by barcode index. Assigns the final `cell_type` column using confidence thresholds:

| Species | Probability column | Name column | Threshold |
|---|---|---|---|
| Human | `supercluster_bootstrapping_probability` | `supercluster_name` | ≥ 0.5 |
| Mouse | `class_bootstrapping_probability` | `class_name` | ≥ 0.5 |

Cells below the probability threshold are assigned `cell_type = "Unknown"`.

If no labels CSV exists (E2 was skipped), all cells receive `cell_type = "Unknown"` and the step still writes a valid output h5ad so downstream steps have a consistent input path.

---

### G — AtlasCleanerStep

**File:** `src/data_processing/atlas_cleaner_step.py`
**Input:** `6.labeled/{id}.h5ad`
**Output:** `7.atlas_clean/{id}.h5ad`, `7.atlas_clean/{id}_clean_result.json`

Prepares a single dataset for atlas inclusion:

1. Validates `gene_ids` column in `.var` (fails if absent)
2. Replaces invalid gene IDs (nan / None / empty) with `gene_symbols`
3. Removes genes whose symbol starts with `"nan-"` (mapping artefacts)
4. Removes cells with `cell_type == "Unknown"`
5. Enforces minimum of **200 cells** after Unknown removal (`status: "skipped"` otherwise)
6. Makes `obs_names` globally unique: `{dataset_id}_{barcode}`
7. Drops scrublet / QC obs columns not needed downstream
8. Clears `.layers`, `.raw`, `.obsm`, `.varm`, `.obsp`, `.uns`
9. Converts `.X` to `float32` CSR

---

### H — AtlasMergerStep

**File:** `src/data_processing/atlas_merger_step.py`
**Input:** `7.atlas_clean/{id}.h5ad` for each dataset ID in the provided list
**Output:** `8.atlas/{atlas_name}.h5ad`, `8.atlas/{atlas_name}_merge_result.json`

Merges all cleaned h5ad files into one atlas:

1. Concatenates with `anndata.concat(join="outer")` — all genes retained, zeros for missing
2. For collections larger than `batch_size=800`, intermediate batch files are saved to avoid OOM
3. Filters genes expressed in fewer than 200 cells
4. Stores raw counts in `.layers["counts"]`
5. Normalises to 10 000 counts per cell (`sc.pp.normalize_total`)
6. log1p transforms
7. Selects top 5 000 highly variable genes (`batch_key="batch"`)
8. Writes final atlas

**Usage:**

```python
from src.data_processing.atlas_merger_step import AtlasMergerStep

step = AtlasMergerStep(data_folder="/path/to/data")
result = step.merge(
    dataset_ids=["39578645_01", "39578645_02", ...],
    atlas_name="human_atlas_v1",
)
```

## External Tool Dependencies

| Tool | Step | Required env var | Notes |
|---|---|---|---|
| Gene mart TSV | B2 | `HUMAN_GENE_MART_PATH`, `MOUSE_GENE_MART_PATH` | tab-sep, columns: `gene_ids`, `gene_symbols` |
| Scrublet | D | — | `pip install scrublet`; auto-installed via `pyproject.toml` |
| cell_type_mapper | E2 | `HUMAN_MAPMYCELLS_TAXONOMY_PATH`, `MOUSE_MAPMYCELLS_TAXONOMY_PATH` | `pip install cell-type-mapper`; step degrades gracefully if absent |
| R / rpy2 | B2 (skip) | — | Datasets with `.rds` / `.rdata` files set `requires_r_extraction=True` and are bypassed by the Python pipeline |

## Resuming Interrupted Runs

Each step guards its own output. To re-run a step from scratch, delete its output file(s) before calling it again.

Example — re-run QC threshold review for one dataset:

```bash
rm $DATA_FOLDER/4.qc/39578645_01_thresholds.json
```

## Tests

| Test file | Steps covered | LLM required |
|---|---|---|
| `system_tests/test_metadata_extractor_step.py` | A0 | Yes (skipped by default) |
| `system_tests/test_repository_analyst_step.py` | A1 | Yes for some tests |
| `system_tests/test_data_downloader.py` | A2 | No |
| `system_tests/test_format_analyzer_step.py` | B1 | Yes (skipped by default) |
| `system_tests/test_format_converter_step.py` | B2 | No |
| `system_tests/test_count_qc_step.py` | C1 | No |
| `system_tests/test_qc_reviewer_step.py` | C2 | Yes (skipped by default) |
| `system_tests/test_qc_filter_step.py` | C3 | No |
| `system_tests/test_doublet_detection_step.py` | D | No |
| `system_tests/test_annotation_steps.py` | E1, E2, F | Yes for one test |
| `system_tests/test_atlas_steps.py` | G, H | No |

Run all non-LLM tests:

```bash
eval $(poetry env activate)
python -m pytest system_tests/
```
