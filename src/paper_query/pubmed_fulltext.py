from __future__ import annotations
import os
import time
from dataclasses import dataclass
import logging
from typing import Dict, Optional, Union
import xml.etree.ElementTree as ET

import requests

from .article_retriever import ArticleRetriever

logger = logging.getLogger(__name__)

MAX_RECAPTCHA_RETRIES = 5
RECAPTCHA_RETRY_DELAY_SECONDS = 2
RECAPTCHA_MARKERS = (
    "recaptcha",
    "g-recaptcha",
    "grecaptcha",
    "are you a robot",
    "checking your browser",
    "cf-challenge",
    "just a moment",
    "captcha-delivery",
)


def _looks_like_recaptcha(content: Optional[Union[str, bytes]]) -> bool:
    if not content:
        return False
    if isinstance(content, bytes):
        try:
            text = content.decode("utf-8", errors="ignore")
        except Exception:
            return False
    else:
        text = content
    lowered = text.lower()
    return any(marker in lowered for marker in RECAPTCHA_MARKERS)


@dataclass(frozen=True)
class FullTextResult:
    pmid: str
    url: str
    code: int = 200
    pmcid: Optional[str] = None
    content_type: Optional[str] = None
    content: Optional[Union[str, bytes]] = None


class PubMedFullTextRetriever:
    """Retrieve PMC full text for PubMed papers (HTML via E-Utility, ArticleRetriever fallback)."""

    def __init__(
        self,
        email: Optional[str] = None,
        tool: str = "biomarker_curator",
        api_key: Optional[str] = None,
        base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        timeout: int = 30,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.email = email or os.getenv("NCBI_EMAIL")
        self.tool = tool
        self.api_key = api_key or os.getenv("NCBI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update(self._default_headers())

    def retrieve(self, pmid: str) -> FullTextResult:
        """Return full text. Try PMC via efetch XML first, fall back to ArticleRetriever."""
        try:
            pmcid = self._find_pmcid(pmid)
        except Exception as exc:
            logger.warning("PMCID lookup failed for %s: %s — falling back to ArticleRetriever", pmid, exc)
            pmcid = None

        if pmcid:
            result = self._fetch_xml_via_efetch(pmid, pmcid)
            if result.code < 400 and result.content:
                return result
            logger.warning(
                "efetch XML failed for %s (%s): code=%s — falling back to ArticleRetriever",
                pmid, pmcid, result.code,
            )

        return self._fetch_html_with_article_retriever_recaptcha_retry(pmid)

    def _fetch_xml_via_efetch(self, pmid: str, pmcid: str) -> FullTextResult:
        pmc_number = pmcid[3:] if pmcid.startswith("PMC") else pmcid
        params: Dict[str, str] = {
            "db": "pmc",
            "id": pmc_number,
            "rettype": "full",
            "retmode": "xml",
        }
        self._add_common_params(params)
        url = f"{self.base_url}/efetch.fcgi"
        response = self.session.get(url, params=params, timeout=self.timeout)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            logger.warning("efetch XML failed for %s (%s): %s", pmid, pmcid, exc)
            return FullTextResult(
                pmid=pmid, pmcid=pmcid, content_type="application/xml",
                content=None, url=url, code=response.status_code,
            )
        content = response.content.decode("utf-8", errors="replace")
        logger.info("PubMedFullTextRetriever: fetched efetch XML for %s", pmcid)
        return FullTextResult(
            pmid=pmid, pmcid=pmcid, content_type="application/xml",
            content=content, url=url, code=response.status_code,
        )

    def _fetch_html_with_article_retriever_recaptcha_retry(self, pmid: str) -> FullTextResult:
        result = self._fetch_html_with_article_retriever(pmid)
        for attempt in range(2, MAX_RECAPTCHA_RETRIES + 1):
            if not _looks_like_recaptcha(result.content):
                return result
            logger.warning(
                "ArticleRetriever HTML for %s looks like a recaptcha/challenge page — retry %d/%d",
                pmid, attempt, MAX_RECAPTCHA_RETRIES,
            )
            time.sleep(RECAPTCHA_RETRY_DELAY_SECONDS)
            result = self._fetch_html_with_article_retriever(pmid)
        return result

    def _fetch_html_with_article_retriever(self, pmid: str) -> FullTextResult:
        retriever = ArticleRetriever()
        res, html_content, code = retriever.request_article(pmid)
        return FullTextResult(
            pmid=pmid,
            content_type="text/html",
            content=html_content,
            url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/pmid/{pmid}/",
            code=code,
        )

    def _find_pmcid(self, pmid: str) -> Optional[str]:
        params: Dict[str, str] = {
            "dbfrom": "pubmed",
            "db": "pmc",
            "id": pmid,
            "retmode": "xml",
        }
        self._add_common_params(params)
        response = self.session.get(
            f"{self.base_url}/elink.fcgi",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._parse_pmcid(response.text)

    @staticmethod
    def _parse_pmcid(xml_text: str) -> Optional[str]:
        root = ET.fromstring(xml_text)
        for linkset in root.findall(".//LinkSetDb"):
            link_name = linkset.findtext("LinkName")
            if link_name not in {"pubmed_pmc", "pubmed_pmc_local"}:
                continue
            id_nodes = linkset.findall("./Link/Id")
            if not id_nodes:
                continue
            pmc_id = id_nodes[0].text
            if not pmc_id:
                continue
            return f"PMC{pmc_id}" if not pmc_id.startswith("PMC") else pmc_id
        return None

    def _add_common_params(self, params: Dict[str, str]) -> None:
        if self.email:
            params["email"] = self.email
        if self.tool:
            params["tool"] = self.tool
        if self.api_key:
            params["api_key"] = self.api_key

    def _default_headers(self) -> Dict[str, str]:
        contact = self.email or "unknown"
        return {
            "User-Agent": f"{self.tool} ({contact})",
            "Accept": "text/html,*/*;q=0.8",
        }
