
import pytest
import logging
from src.paper_query.pubmed_fulltext import PubMedFullTextRetriever, FullTextResult

logger = logging.getLogger(__name__)

PMIDS = [
    "33083725",
    "36170369",
    "39554066",
    "39314329",
    "37558676",
    "38325728",
    "40329537",
    "40060644",
    "39543407",
    "37963245",
    "39329069",
]


@pytest.fixture(scope="module")
def retriever():
    return PubMedFullTextRetriever()


@pytest.mark.skip()
@pytest.mark.parametrize("pmid", PMIDS)
def test_find_pmcid(retriever, pmid):
    """PMCID lookup via E-Utility elink."""
    pmcid = retriever._find_pmcid(pmid)
    if pmcid:
        logger.info("PMID %s -> PMCID %s", pmid, pmcid)
        assert pmcid.startswith("PMC")
    else:
        logger.warning("PMID %s: no PMCID found (not in PMC)", pmid)


@pytest.mark.skip()
@pytest.mark.parametrize("pmid", PMIDS)
def test_retrieve_returns_html_content(retriever, pmid):
    """retrieve() returns non-empty HTML content for every PMID."""
    result = retriever.retrieve(pmid)

    assert isinstance(result, FullTextResult)
    assert result.pmid == pmid
    assert result.code < 400, f"PMID {pmid}: HTTP {result.code}"
    assert result.content, f"PMID {pmid}: content is empty"
    assert result.content_type == "text/html", (
        f"PMID {pmid}: expected text/html, got {result.content_type}"
    )

    path = "E-Utility PMC HTML" if result.pmcid else "ArticleRetriever fallback"
    logger.info(
        "PMID %s | path=%s | pmcid=%s | code=%s | content_len=%d",
        pmid, path, result.pmcid, result.code, len(result.content),
    )


# @pytest.mark.skip()
@pytest.mark.parametrize("pmid", PMIDS)
def test_retrieve_html_contains_key_sections(retriever, pmid):
    """Retrieved HTML should contain at least one of the expected full-text sections."""
    result = retriever.retrieve(pmid)

    assert result.content, f"PMID {pmid}: no content returned"
    content_lower = (
        result.content.lower()
        if isinstance(result.content, str)
        else result.content.decode("utf-8", errors="ignore").lower()
    )

    has_methods = "methods" in content_lower
    has_abstract = "abstract" in content_lower
    if not has_methods:
        logger.warning("PMID %s: no 'methods' section found in HTML", pmid)
    if not has_abstract:
        logger.warning("PMID %s: no 'abstract' found in HTML", pmid)

    assert has_abstract or has_methods, (
        f"PMID {pmid}: HTML appears incomplete — neither 'abstract' nor 'methods' found"
    )

    # let's write the content to a html file
    with open(f"./system_tests/data/test_pubmed_fulltext_{pmid}.html", "w") as f:
        f.write(result.content)
    logger.info("PMID %s: content written to test_pubmed_fulltext_%s.html", pmid, pmid)
