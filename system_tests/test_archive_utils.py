"""
Offline tests for archive extraction + input-path resolution (Pipeline 2).

No network: archives and 10x fixtures are built in tmp_path.
"""

import gzip
import os
import tarfile
import zipfile

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from src.data_processing.archive_utils import (
    archive_stem,
    extract_archive,
    find_10x_dir,
    find_all_10x_dirs,
    is_archive,
    resolve_input_path,
    resolve_matrix_inputs,
    sample_name_from_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_10x_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    matrix = sp.csr_matrix(np.array([[1, 0, 3, 0], [0, 5, 0, 7], [2, 2, 2, 2]], dtype=float))
    mtx_path = os.path.join(path, "matrix.mtx")
    sio.mmwrite(mtx_path, matrix)
    with open(mtx_path, "rb") as fin, gzip.open(mtx_path + ".gz", "wb") as fout:
        fout.write(fin.read())
    os.remove(mtx_path)
    with gzip.open(os.path.join(path, "barcodes.tsv.gz"), "wt") as f:
        f.write("\n".join(["ACGT-1", "TGCA-2", "GCTA-3", "ATCG-4"]) + "\n")
    with gzip.open(os.path.join(path, "features.tsv.gz"), "wt") as f:
        f.write("ENSG001\tGeneA\tGene Expression\nENSG002\tGeneB\tGene Expression\nENSG003\tGeneC\tGene Expression\n")


def _make_tar(src_dir: str, tar_path: str, arcname: str) -> None:
    with tarfile.open(tar_path, "w") as tf:
        tf.add(src_dir, arcname=arcname)


# ---------------------------------------------------------------------------
# is_archive / archive_stem
# ---------------------------------------------------------------------------

def test_is_archive():
    assert is_archive("GSE147528_RAW.tar")
    assert is_archive("data.tar.gz")
    assert is_archive("x.zip")
    assert not is_archive("matrix.mtx.gz")
    assert not is_archive("counts.csv")


def test_archive_stem():
    assert archive_stem("GSE147528_RAW.tar") == "GSE147528_RAW"
    assert archive_stem("data.tar.gz") == "data"
    assert archive_stem("x.zip") == "x"
    assert archive_stem("notarchive.csv") == "notarchive.csv"


# ---------------------------------------------------------------------------
# extract_archive
# ---------------------------------------------------------------------------

def test_extract_tar_then_idempotent(tmp_path):
    src = os.path.join(str(tmp_path), "src", "tenx")
    _make_10x_dir(src)
    tar_path = os.path.join(str(tmp_path), "GSE_RAW.tar")
    _make_tar(src, tar_path, arcname="tenx")

    dest = os.path.join(str(tmp_path), "out")
    members = extract_archive(tar_path, dest)
    assert any(m.endswith("matrix.mtx.gz") for m in members)
    assert any(m.endswith("barcodes.tsv.gz") for m in members)
    assert any(m.endswith("features.tsv.gz") for m in members)

    # Second call must not re-extract; returns the same listing.
    again = extract_archive(tar_path, dest)
    assert again == members


def test_extract_zip(tmp_path):
    src = os.path.join(str(tmp_path), "src", "tenx")
    _make_10x_dir(src)
    zip_path = os.path.join(str(tmp_path), "data.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in os.listdir(src):
            zf.write(os.path.join(src, f), arcname=os.path.join("tenx", f))
    dest = os.path.join(str(tmp_path), "out")
    members = extract_archive(zip_path, dest)
    assert any(m.endswith("matrix.mtx.gz") for m in members)


# ---------------------------------------------------------------------------
# find_10x_dir / resolve_input_path
# ---------------------------------------------------------------------------

def test_find_10x_dir_nested(tmp_path):
    nested = os.path.join(str(tmp_path), "GSE_RAW", "tenx")
    _make_10x_dir(nested)
    found = find_10x_dir(str(tmp_path))
    assert found is not None
    assert os.path.samefile(found, nested)


def test_resolve_10x_ignores_bogus_primary_file(tmp_path):
    """Reproduces the reported bug: primary_file is a description, not a path."""
    dataset_dir = str(tmp_path)
    _make_10x_dir(os.path.join(dataset_dir, "GSE147528_RAW", "tenx"))
    bogus = "GSE147528_RAW.tar -> extracted 10x matrix directory (matrix.mtx/...)"
    resolved = resolve_input_path(dataset_dir, "10x", bogus)
    assert resolved is not None
    assert os.path.isdir(resolved)
    assert os.path.exists(os.path.join(resolved, "matrix.mtx.gz"))


def test_resolve_literal_csv(tmp_path):
    dataset_dir = str(tmp_path)
    with open(os.path.join(dataset_dir, "counts.csv"), "w") as f:
        f.write("a,b\n1,2\n")
    resolved = resolve_input_path(dataset_dir, "csv", "counts.csv")
    assert resolved == os.path.join(dataset_dir, "counts.csv")


def test_resolve_by_basename_anywhere(tmp_path):
    dataset_dir = str(tmp_path)
    sub = os.path.join(dataset_dir, "GSE_RAW")
    os.makedirs(sub)
    target = os.path.join(sub, "expr.h5ad")
    open(target, "w").close()
    resolved = resolve_input_path(dataset_dir, "h5ad", "expr.h5ad")
    assert resolved == target


def test_resolve_never_returns_archive(tmp_path):
    dataset_dir = str(tmp_path)
    open(os.path.join(dataset_dir, "GSE_RAW.tar"), "w").close()
    # data_type csv but only a .tar present → nothing valid to resolve to.
    assert resolve_input_path(dataset_dir, "csv", "GSE_RAW.tar") is None


def test_resolve_missing_returns_none(tmp_path):
    assert resolve_input_path(str(tmp_path), "csv", "nope.csv") is None


# ---------------------------------------------------------------------------
# Multi-sample resolution
# ---------------------------------------------------------------------------

def test_resolve_matrix_inputs_multi_h5(tmp_path):
    """Multiple per-sample .h5 files (GSE147528-style) → multi-sample list."""
    d = str(tmp_path)
    sub = os.path.join(d, "GSE147528_RAW")
    os.makedirs(sub)
    for n in (
        "GSM1_SFG2_raw_gene_bc_matrices_h5.h5",
        "GSM2_SFG1_raw_gene_bc_matrices_h5.h5",
        "GSM3_EC1_raw_gene_bc_matrices_h5.h5",
    ):
        open(os.path.join(sub, n), "w").close()
    bogus = "GSE147528_RAW.tar -> extracted 10x matrix directory"
    inputs = resolve_matrix_inputs(d, "h5", bogus)
    assert len(inputs) == 3
    assert all(p.endswith(".h5") for p in inputs)


def test_resolve_matrix_inputs_csv_is_single(tmp_path):
    """csv siblings (likely metadata) must NOT be treated as multi-sample."""
    d = str(tmp_path)
    open(os.path.join(d, "counts.csv"), "w").close()
    open(os.path.join(d, "metadata.csv"), "w").close()
    inputs = resolve_matrix_inputs(d, "csv", "counts.csv")
    assert inputs == [os.path.join(d, "counts.csv")]


def test_resolve_matrix_inputs_single_h5(tmp_path):
    d = str(tmp_path)
    open(os.path.join(d, "filtered_feature_bc_matrix.h5"), "w").close()
    inputs = resolve_matrix_inputs(d, "h5", "filtered_feature_bc_matrix.h5")
    assert inputs == [os.path.join(d, "filtered_feature_bc_matrix.h5")]


def test_find_all_10x_dirs(tmp_path):
    for sample in ("s1", "s2"):
        sub = os.path.join(str(tmp_path), sample)
        os.makedirs(sub)
        for f in ("matrix.mtx.gz", "barcodes.tsv.gz", "features.tsv.gz"):
            open(os.path.join(sub, f), "w").close()
    dirs = find_all_10x_dirs(str(tmp_path))
    assert len(dirs) == 2


def test_sample_name_from_path():
    assert sample_name_from_path(
        "/x/GSE147528_RAW/GSM4432635_SFG2_raw_gene_bc_matrices_h5.h5"
    ) == "GSM4432635_SFG2"
    assert sample_name_from_path("/x/sampleA.h5") == "sampleA"
    assert sample_name_from_path("/x/GSE_RAW") == "GSE_RAW"  # directory name
