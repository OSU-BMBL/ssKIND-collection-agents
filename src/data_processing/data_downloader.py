"""
Pipeline 2, Step A2 — DataDownloaderStep

Input  : data/1.manifest/{dataset_id}.json
Output : data/2.raw/{dataset_id}/{filename}   (one file per entry in manifest)
         data/2.raw/{dataset_id}/download_status.json

Skips individual files that already exist with matching size.
Uses streaming HTTP with a Content-Length check for partial-download detection.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

import requests

logger = logging.getLogger(__name__)

MANIFEST_SUBDIR = "1.manifest"
RAW_OUTPUT_SUBDIR = "2.raw"
STATUS_FILENAME = "download_status.json"
CHUNK_SIZE = 1024 * 1024  # 1 MB


class DataDownloaderStep:
    """
    Pipeline 2, Step A2: download files listed in a dataset's manifest.

    Input  : data/1.manifest/{dataset_id}.json
    Output : data/2.raw/{dataset_id}/
    """

    def __init__(
        self,
        data_folder: Optional[str] = None,
        timeout: int = 60,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.data_folder = data_folder or os.getenv("DATA_FOLDER", ".")
        self.timeout = timeout
        self.session = session or requests.Session()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, dataset_id: str) -> Optional[dict]:
        """
        Download all files for dataset_id.

        Returns the download-status dict, or None if the manifest is missing.
        Already-completed downloads (all files present and sized correctly)
        return the cached status without re-downloading.
        """
        manifest = self._load_manifest(dataset_id)
        if manifest is None:
            logger.error("DataDownloaderStep: manifest not found for %s", dataset_id)
            return None

        out_dir = self._dataset_dir(dataset_id)
        os.makedirs(out_dir, exist_ok=True)

        status_path = os.path.join(out_dir, STATUS_FILENAME)
        if os.path.exists(status_path):
            cached = self._load_status(status_path)
            if cached.get("all_success"):
                logger.info("DataDownloaderStep: %s already complete, skipping", dataset_id)
                return cached

        file_records = []
        all_success = True

        for entry in manifest.get("files", []):
            record = self._download_file(entry, out_dir)
            file_records.append(record)
            if record["status"] == "failed":
                all_success = False

        status = {
            "dataset_id": dataset_id,
            "downloaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "files": file_records,
            "all_success": all_success,
        }
        with open(status_path, "w") as f:
            json.dump(status, f, indent=2)

        if all_success:
            logger.info("DataDownloaderStep: %s — all %d file(s) downloaded", dataset_id, len(file_records))
        else:
            failed = [r["filename"] for r in file_records if r["status"] == "failed"]
            logger.warning("DataDownloaderStep: %s — failed: %s", dataset_id, failed)

        return status

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _manifest_path(self, dataset_id: str) -> str:
        return os.path.join(self.data_folder, MANIFEST_SUBDIR, f"{dataset_id}.json")

    def _dataset_dir(self, dataset_id: str) -> str:
        return os.path.join(self.data_folder, RAW_OUTPUT_SUBDIR, dataset_id)

    def _load_manifest(self, dataset_id: str) -> Optional[dict]:
        path = self._manifest_path(dataset_id)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _load_status(path: str) -> dict:
        with open(path) as f:
            return json.load(f)

    def _download_file(self, entry: dict, out_dir: str) -> dict:
        """Download one file entry; return a status record."""
        filename = entry["filename"]
        url = entry["url"]
        local_path = os.path.join(out_dir, filename)

        record: dict = {
            "filename": filename,
            "url": url,
            "local_path": local_path,
            "status": "pending",
            "size_bytes": None,
            "error": None,
        }

        # Check if already fully downloaded
        if os.path.exists(local_path):
            expected_size = self._remote_size(url)
            local_size = os.path.getsize(local_path)
            if expected_size is not None and local_size == expected_size:
                logger.info("DataDownloaderStep: skipping %s (already complete)", filename)
                record["status"] = "skipped"
                record["size_bytes"] = local_size
                return record
            if expected_size is not None and local_size != expected_size:
                logger.warning(
                    "DataDownloaderStep: %s size mismatch (local=%d, remote=%d) — re-downloading",
                    filename, local_size, expected_size,
                )

        # Stream download
        try:
            logger.info("DataDownloaderStep: downloading %s from %s", filename, url)
            with self.session.get(url, stream=True, timeout=self.timeout) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                with open(local_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            fh.write(chunk)
                            downloaded += len(chunk)
            record["status"] = "success"
            record["size_bytes"] = downloaded
            logger.info(
                "DataDownloaderStep: %s done (%.1f MB)",
                filename, downloaded / 1024 / 1024,
            )
        except Exception as exc:
            logger.error("DataDownloaderStep: failed to download %s: %s", filename, exc)
            record["status"] = "failed"
            record["error"] = str(exc)

        return record

    def _remote_size(self, url: str) -> Optional[int]:
        """Return Content-Length from a HEAD request, or None if unavailable."""
        try:
            resp = self.session.head(url, timeout=10, allow_redirects=True)
            resp.raise_for_status()
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl else None
        except Exception:
            return None
