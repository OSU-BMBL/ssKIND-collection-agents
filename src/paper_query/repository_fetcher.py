"""Helpers to fetch file listings from public genomics repositories (GEO, SRA, Zenodo)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RepoFile:
    filename: str
    url: str
    size: Optional[str] = None   # human-readable, e.g. "3.9G"


@dataclass
class RepoListing:
    accession_id: str
    listing_url: str
    files: List[RepoFile]
    error: Optional[str] = None   # non-None when fetch failed


# ---------------------------------------------------------------------------
# GEO helpers
# ---------------------------------------------------------------------------

def _geo_suppl_url(accession_id: str) -> str:
    """
    Build the HTTPS URL for a GEO series supplementary file directory.

    Pattern: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE{prefix}nnn/{accession}/suppl/
    where prefix = all digits except the last three.
    """
    digits = accession_id[3:]  # strip "GSE"
    prefix = (digits[:-3] if len(digits) > 3 else "") + "nnn"
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE{prefix}/{accession_id}/suppl/"


def _parse_ftp_index(html: str, base_url: str) -> List[RepoFile]:
    """Parse an Apache/NCBI directory-listing HTML page into RepoFile list."""
    soup = BeautifulSoup(html, "html.parser")
    files: List[RepoFile] = []
    pre = soup.find("pre")
    if not pre:
        return files
    for a in pre.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/") or href.startswith("?") or "Parent" in a.text:
            continue
        # Extract size from the surrounding text (appears after the date stamp)
        line = a.parent.get_text() if a.parent else ""
        size_match = re.search(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+(\S+)", line)
        size = size_match.group(1) if size_match else None
        url = base_url.rstrip("/") + "/" + href.lstrip("/")
        files.append(RepoFile(filename=href, url=url, size=size))
    return files


def fetch_geo_suppl_listing(
    accession_id: str,
    session: Optional[requests.Session] = None,
    timeout: int = 20,
) -> RepoListing:
    """
    Fetch the supplementary file listing for a GEO series accession (GSE…).
    Returns a RepoListing; .error is set if the request failed.
    """
    listing_url = _geo_suppl_url(accession_id)
    sess = session or requests.Session()
    try:
        resp = sess.get(listing_url, timeout=timeout)
        resp.raise_for_status()
        files = _parse_ftp_index(resp.text, listing_url)
        logger.info(
            "fetch_geo_suppl_listing: %s → %d file(s) at %s",
            accession_id, len(files), listing_url,
        )
        return RepoListing(
            accession_id=accession_id,
            listing_url=listing_url,
            files=files,
        )
    except Exception as exc:
        logger.warning("fetch_geo_suppl_listing failed for %s: %s", accession_id, exc)
        return RepoListing(
            accession_id=accession_id,
            listing_url=listing_url,
            files=[],
            error=str(exc),
        )


def search_geo_by_pmid(
    pmid: str,
    email: Optional[str] = None,
    session: Optional[requests.Session] = None,
    timeout: int = 15,
) -> List[str]:
    """
    Search NCBI GEO (db=gds) for series linked to a PubMed ID.
    Returns a list of GSE accession IDs (may be empty).
    """
    sess = session or requests.Session()
    params: dict = {
        "db": "gds",
        "term": f"{pmid}[PMID] AND GSE[ETYP]",
        "retmode": "json",
        "retmax": "20",
    }
    if email:
        params["email"] = email
    try:
        resp = sess.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params=params,
            timeout=timeout,
        )
        resp.raise_for_status()
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        # Convert GDS UIDs → GSE accession IDs via esummary
        summary_resp = sess.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "gds", "id": ",".join(ids), "retmode": "json"},
            timeout=timeout,
        )
        summary_resp.raise_for_status()
        result = summary_resp.json().get("result", {})
        accessions = []
        for uid in ids:
            entry = result.get(uid, {})
            acc = entry.get("accession", "")
            if acc.startswith("GSE"):
                accessions.append(acc)
        logger.info("search_geo_by_pmid: PMID %s → %s", pmid, accessions)
        return accessions
    except Exception as exc:
        logger.warning("search_geo_by_pmid failed for PMID %s: %s", pmid, exc)
        return []
