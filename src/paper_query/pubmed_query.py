from typing import Any
import requests
import logging 
import time
import math
import xml.etree.ElementTree as ET

from src.database.pmid_paper_db import PMIDPaperDB
from .article_retriever import ArticleRetriever

logger = logging.getLogger(__name__)

PUBMED_DATABASE_LIST = [
    "pubmed",
    "protein",
    "nuccore",
    "ipg",
    "nucleotide",
    "structure",
    "genome",
    "annotinfo",
    "assembly",
    "bioproject",
    "biosample",
    "blastdbinfo",
    "books",
    "cdd",
    "clinvar",
    "gap",
    "gapplus",
    "grasp",
    "dbvar",
    "gene",
    "gds",
    "geoprofiles",
    "medgen",
    "mesh",
    "nlmcatalog",
    "omim",
    "orgtrack",
    "pmc",
    "proteinclusters",
    "pcassay",
    "protfam",
    "pccompound",
    "pcsubstance",
    "seqannot",
    "snp",
    "sra",
    "taxonomy",
    "biocollections",
    "gtr"
]
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def build_query_param(
    mindate: str | None = None,
    maxdate: str | None = None,
    datetype: str = "pdat",
) -> dict:
    mindate_dict = {} if mindate is None else {"mindate": mindate}
    maxdate_dict = {} if maxdate is None else {"maxdate": maxdate}
    datetype_dict = {} if mindate is None and maxdate is None \
        else {"datetype": datetype if datetype is not None else "pdat"}
    return {**mindate_dict, **maxdate_dict, **datetype_dict}

def safe_get(url, params, delay=0.4):
    time.sleep(delay)  # ~3 requests/second
    return requests.get(url, params=params)

def safe_int(s, default=0):
    try:
        return int(s)
    except (ValueError, TypeError):
        logger.error("Error ocurred in converting {s} to int")
        return default

def query_count(
    term: str,
    mindate: str | None = None,
    maxdate: str | None = None,
    datetype: str = "pdat",
) -> int:
    
    query_param = build_query_param(
        mindate=mindate,
        maxdate=maxdate,
        datetype=datetype,
    )
    params = {
        "term": term,
        "retmode": "json",
        "retmax": 0,
        **query_param,
    }
    dbs = ["pubmed"]
    for db in dbs:
        res = None
        try:
            result = safe_get(
                url=ESEARCH_URL,
                params={
                    **params,
                    "db": db,
                }
            )
            res: Any = result.json()
            cnt = safe_int(res['esearchresult']['count'], 0)
            return cnt
        except Exception as e:
            error_in_res = res['error'] if res and 'error' in res else 'unknown error'
            logger.error(f"Error occurred in querying {db}: str(e)\n Error: {error_in_res}")
            return 0
    return 0

STEP_COUNT = 100    
def query_pmids(
    term: str,
    count: int,
    mindate: str | None = None,
    maxdate: str | None = None,
    datetype: str = "pdat",
):
    query_params = build_query_param(
        mindate=mindate,
        maxdate=maxdate,
        datetype=datetype
    )
    params = {
        "term": term,
        "retmode": "json",
        "retmax": STEP_COUNT,
        "db": "pubmed",
        **query_params,
    }
    for ix in range(math.ceil(count/STEP_COUNT)):
        ids = []
        try:
            result = safe_get(
                url=ESEARCH_URL,
                params= {
                    **params,
                    "retstart": ix*STEP_COUNT,
                }
            )
            res =result.json()
            ids = res['esearchresult']['idlist']
        except Exception as e:
            logger.error(str(e))
        for id in ids:
            yield id
    
def query_title_abstract_ispreprint(pmid: str) -> tuple[str | None, str | None, bool]:
    """Queries the title, abstract, and preprint status of a paper by its PubMed ID (PMID).
    Args:
        pmid (str): The PubMed ID of the paper.
    Returns:
        tuple: A tuple containing the title, abstract, and a boolean indicating if it is a preprint.
    Raises:
        Exception: If there is an error during the request or parsing the response.
    """
    params = {
        "id": pmid,
        "db": "pubmed",
        "retmode": "xml"
    }
    try:
        result = safe_get(
            url=EFETCH_URL,
            params=params,
        )
        root = ET.fromstring(result.content)
        is_preprint = False
        for article in root.findall(".//PubmedArticle"):
            title = article.findtext(".//ArticleTitle")
            abstract = article.findtext(".//Abstract/AbstractText")
            publication_types = article.findall(".//PublicationType")
            for pub_type in publication_types:
                if pub_type.text.lower() in ["preprint", "pre-print"]:
                    is_preprint = True
            return title, abstract, is_preprint
        
        return None, None, False
    except Exception as e:
        logger.error(str(e))
        return None, None, False
    
def query_full_text(pmid: str) -> tuple[bool, str | None]:
    """
    Queries the full text of a paper by its PubMed ID (PMID).
    Args:
        pmid (str): The PubMed ID of the paper.
    Returns:
        tuple: A tuple containing a boolean indicating success and the full text content (html format) or None if not found.
    Raises:
        Exception: If there is an error during the request.
    """
    retriever = ArticleRetriever()
    res, html_content, _ =retriever.request_article(pmid)

    return res, html_content


class PubMedPaperRetriever:
    """
    A class to retrieve PubMed papers based on a search term, and to fetch their titles, abstracts, and full text.
    """

    def __init__(self):
        self.db = PMIDPaperDB()
    
    def query_pmids(
        self,
        term: str,
        count: int,
        mindate: str | None = None,
        maxdate: str | None = None,
        datetype: str = "pdat",
    ):
        """
        Queries PubMed for PMIDs based on a search term and date range.
        
        Args:
            term (str): The search term to query PubMed.
            count (int): The number of PMIDs to retrieve.
            mindate (str, optional): The minimum date for the search.
            maxdate (str, optional): The maximum date for the search.
            datetype (str): The type of date to use for the search.
        
        Yields:
            str: A PMID from the search results.
        """
        return query_pmids(term, count, mindate, maxdate, datetype)
    
    def query_title_abstract_ispreprint(self, pmid: str) -> tuple[str | None, str | None, bool]:
        """
        Queries the title, abstract, and preprint status of a paper by its PubMed ID (PMID).
        
        Args:
            pmid (str): The PubMed ID of the paper.
        
        Returns:
            tuple: A tuple containing the title, abstract, and a boolean indicating if it is a preprint.
        """
        title, abstract, is_preprint = self.db.select_paper_title_abstract(pmid)
        if title is not None:
            return title, abstract, is_preprint
        title, abstract, is_preprint = query_title_abstract_ispreprint(pmid)
        self.db.insert_paper_title_abstract(pmid, title, abstract, is_preprint)
        return title, abstract, is_preprint
    
    def query_full_text(self, pmid: str) -> tuple[bool, str | None]:
        """
        Queries the full text of a paper by its PubMed ID (PMID).
        
        Args:
            pmid (str): The PubMed ID of the paper.
        
        Returns:
            tuple: A tuple containing a boolean indicating success and the full text content (html format) or None if not found.
        """
        html_content = self.db.select_paper_html_content(pmid)
        if html_content is not None:
            return True, html_content
        res, html_content = query_full_text(pmid)
        if res and html_content:
            self.db.insert_paper_html_content(pmid, html_content)
        return res, html_content


