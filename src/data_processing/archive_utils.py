"""
Archive extraction + input-path resolution helpers for Pipeline 2.

Kept dependency-light (stdlib only) so both the agent steps (B1) and the
programmatic steps (A2, B2) can import it without pulling in scanpy/anndata.

Two responsibilities:
  * Unpack downloaded archives (GEO `*_RAW.tar`, `.tar.gz`, `.zip`, …) so the
    real count-matrix files exist on disk for later steps.
  * Resolve a conversion config's `primary_file` to a concrete path on disk —
    auto-locating a 10x directory or a file of the right extension when the
    recorded value is missing or only a description.
"""

from __future__ import annotations

import logging
import os
import re
import tarfile
import zipfile
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

# Longest-suffix-first so `_archive_stem` strips the full extension.
ARCHIVE_EXTENSIONS = (".tar.gz", ".tar.bz2", ".tgz", ".tbz2", ".tar", ".zip")

# data_type → file extensions that can satisfy it (used when auto-locating).
_DATA_TYPE_EXTENSIONS = {
    "csv": (".csv",),
    "csv.gz": (".csv.gz",),
    "tsv": (".tsv",),
    "tsv.gz": (".tsv.gz",),
    "txt": (".txt",),
    "mtx": (".mtx", ".mtx.gz"),
    "10x_matrix": (".mtx", ".mtx.gz"),
    "h5": (".h5",),
    "h5ad": (".h5ad",),
    "rds": (".rds", ".rds.gz"),
    "rdata": (".rdata", ".rda"),
}


def is_archive(filename: str) -> bool:
    low = filename.lower()
    return any(low.endswith(ext) for ext in ARCHIVE_EXTENSIONS)


def archive_stem(filename: str) -> str:
    """`GSE1_RAW.tar.gz` → `GSE1_RAW` (drops the archive extension)."""
    low = filename.lower()
    for ext in ARCHIVE_EXTENSIONS:
        if low.endswith(ext):
            return filename[: len(filename) - len(ext)]
    return filename


def list_files(root: str) -> List[str]:
    """Sorted list of file paths under `root`, relative to `root`."""
    out: List[str] = []
    if not os.path.isdir(root):
        return out
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            out.append(os.path.relpath(os.path.join(dirpath, f), root))
    return sorted(out)


def _is_within(base: str, target: str) -> bool:
    base = os.path.realpath(base)
    target = os.path.realpath(target)
    return os.path.commonpath([base, target]) == base


def extract_archive(archive_path: str, dest_dir: str) -> List[str]:
    """Extract a tar/zip archive into `dest_dir`; return member file paths.

    Idempotent: if `dest_dir` already contains files, the existing listing is
    returned without re-extracting. Guards against path-traversal entries.
    """
    os.makedirs(dest_dir, exist_ok=True)
    existing = list_files(dest_dir)
    if existing:
        logger.info("extract_archive: %s already extracted (%d files)", dest_dir, len(existing))
        return existing

    low = archive_path.lower()
    if low.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            for name in zf.namelist():
                if not _is_within(dest_dir, os.path.join(dest_dir, name)):
                    raise ValueError(f"Unsafe path in zip archive: {name!r}")
            zf.extractall(dest_dir)
    else:
        with tarfile.open(archive_path, "r:*") as tf:
            try:
                tf.extractall(dest_dir, filter="data")  # Python 3.12+ safe filter
            except TypeError:
                for member in tf.getmembers():
                    if not _is_within(dest_dir, os.path.join(dest_dir, member.name)):
                        raise ValueError(f"Unsafe path in tar archive: {member.name!r}")
                tf.extractall(dest_dir)

    members = list_files(dest_dir)
    logger.info("extract_archive: %s → %s (%d files)", archive_path, dest_dir, len(members))
    return members


def _dir_has_10x_triplet(files_lower: Sequence[str]) -> bool:
    has_matrix = any(f in ("matrix.mtx", "matrix.mtx.gz") for f in files_lower)
    has_barcodes = any("barcodes" in f for f in files_lower)
    has_features = any(("features" in f) or ("genes" in f) for f in files_lower)
    return has_matrix and has_barcodes and has_features


def find_10x_dir(root: str) -> Optional[str]:
    """Return the directory under `root` holding a 10x matrix, or None.

    Prefers a directory with the canonical triplet (matrix.mtx[.gz] +
    barcodes + features/genes); falls back to any directory that contains a
    `*matrix.mtx[.gz]` file (e.g. GEO per-sample prefixed members).
    """
    fallback = None
    if not os.path.isdir(root):
        return None
    for dirpath, _dirs, files in os.walk(root):
        lower = [f.lower() for f in files]
        if _dir_has_10x_triplet(lower):
            return dirpath
        if fallback is None and any(
            f.endswith("matrix.mtx") or f.endswith("matrix.mtx.gz") for f in lower
        ):
            fallback = dirpath
    return fallback


def find_all_10x_dirs(root: str) -> List[str]:
    """Every directory under `root` that holds a canonical 10x triplet."""
    dirs: List[str] = []
    if not os.path.isdir(root):
        return dirs
    for dirpath, _dirs, files in os.walk(root):
        if _dir_has_10x_triplet([f.lower() for f in files]):
            dirs.append(dirpath)
    return sorted(dirs)


def _all_files_with_ext(root: str, exts: Sequence[str]) -> List[str]:
    out: List[str] = []
    if not os.path.isdir(root):
        return out
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if is_archive(f):
                continue
            low = f.lower()
            if any(low.endswith(e) for e in exts):
                out.append(os.path.join(dirpath, f))
    return sorted(out)


# Formats where multiple sibling files mean multiple samples to merge. csv/tsv
# are excluded: their siblings are typically metadata, not extra samples.
_MULTI_SAMPLE_EXTENSIONS = {
    "h5": (".h5",),
    "mtx": (".mtx", ".mtx.gz"),
    "10x_matrix": (".mtx", ".mtx.gz"),
}

_SAMPLE_STRIP_RE = re.compile(r"_(raw|filtered)[_-].*$", re.IGNORECASE)
_KNOWN_INPUT_EXTS = (
    ".h5ad", ".h5", ".csv.gz", ".tsv.gz", ".mtx.gz", ".rds.gz",
    ".csv", ".tsv", ".txt", ".mtx", ".rds", ".rdata", ".rda",
)


def sample_name_from_path(path: str) -> str:
    """Derive a sample label from a per-sample matrix path.

    `GSM4432635_SFG2_raw_gene_bc_matrices_h5.h5` → `GSM4432635_SFG2`.
    For a directory, uses the directory name.
    """
    name = os.path.basename(os.path.normpath(path))
    low = name.lower()
    for ext in _KNOWN_INPUT_EXTS:
        if low.endswith(ext):
            name = name[: len(name) - len(ext)]
            break
    stripped = _SAMPLE_STRIP_RE.sub("", name)
    return stripped or name


def resolve_matrix_inputs(
    dataset_dir: str, data_type: str, primary_file: Optional[str]
) -> List[str]:
    """Return the matrix inputs to convert for a dataset.

    More than one element means a multi-sample dataset (e.g. one Cell Ranger
    `.h5` per GEO sample) that the converter should merge into a single h5ad.
    csv/tsv/h5ad/rds resolve to a single file. Returns [] if nothing matches.
    """
    dt = (data_type or "").lower().strip()

    if dt == "10x":
        dirs = find_all_10x_dirs(dataset_dir)
        if dirs:
            return dirs

    exts = _MULTI_SAMPLE_EXTENSIONS.get(dt)
    if exts:
        files = _all_files_with_ext(dataset_dir, exts)
        if len(files) > 1:
            return files

    single = resolve_input_path(dataset_dir, data_type, primary_file)
    return [single] if single else []


def _largest_file_with_ext(root: str, exts: Sequence[str]) -> Optional[str]:
    best, best_size = None, -1
    if not os.path.isdir(root):
        return None
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if is_archive(f):
                continue
            low = f.lower()
            if any(low.endswith(e) for e in exts):
                path = os.path.join(dirpath, f)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if size > best_size:
                    best, best_size = path, size
    return best


def resolve_input_path(
    dataset_dir: str, data_type: str, primary_file: Optional[str]
) -> Optional[str]:
    """Resolve a converter input to a real path under `dataset_dir`.

    Resolution order:
      1. For 10x: locate the matrix directory (the recorded primary_file is
         often a directory name or — when an archive wasn't unpacked — a
         description, so the directory scan is authoritative).
      2. The literal primary_file, as a path or by basename anywhere in the
         tree (archives are never accepted as a converter input).
      3. The largest file matching the data_type's extension.

    Returns an absolute path (file or directory), or None if nothing matches.
    """
    dt = (data_type or "").lower().strip()

    # 1. 10x directories are unambiguous — find the triplet on disk.
    if dt == "10x":
        found = find_10x_dir(dataset_dir)
        if found:
            return found

    # 2. Literal primary_file (strip any "name -> description" annotation).
    if primary_file:
        raw = primary_file.split("->")[0].strip()
        if raw:
            candidate = raw if os.path.isabs(raw) else os.path.join(dataset_dir, raw)
            if os.path.exists(candidate) and not is_archive(candidate):
                return candidate
            base = os.path.basename(raw)
            if base and not is_archive(base):
                for dirpath, _dirs, files in os.walk(dataset_dir):
                    if base in files:
                        return os.path.join(dirpath, base)

    # 3. Auto-locate by extension for the declared data_type.
    exts = _DATA_TYPE_EXTENSIONS.get(dt)
    if exts:
        found = _largest_file_with_ext(dataset_dir, exts)
        if found:
            return found

    return None
