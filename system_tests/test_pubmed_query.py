
import pytest
import logging
from src.paper_query.pubmed_query import (
    query_count, 
    query_pmids, 
    query_title_abstract_ispreprint, 
    query_full_text,
)
from src.paper_query.html_extractor import HtmlTableExtractor
from src.workflow.workflow_utils import convert_html_to_plaintext

logger = logging.getLogger(__name__)

@pytest.mark.skip()
def test_query_count():
    query = "(Alzheimer[ti] OR Alzheimer[ab]) AND (single[ab] OR single[ti]) AND (RNA OR RNA-seq OR snRNA OR scRNA)"
    cnt = query_count(
        query,
        mindate="2024/06/01",
        maxdate="2025/06/01",
        datetype="pdat",
    )
    print(cnt)

@pytest.mark.skip()
def test_query_pmids():
    # query = "Alzheimer[ti] OR Alzheimer[ab]"
    query = "(Alzheimer[ti] OR Alzheimer[ab]) AND (single[ab] OR single[ti]) AND (RNA OR RNA-seq OR snRNA OR scRNA)"
    pmid_cnt = query_count(
        query,
        mindate="2024/06/01",
        maxdate="2025/06/01",
        datetype="pdat",
    )
    ids = query_pmids(
        query,
        pmid_cnt,
        mindate="2024/06/01",
        maxdate="2025/06/01",
        datetype="pdat",
    )
    pmids = list(ids)
    for id in pmids:
        logger.info(id)
    assert pmids is not None and len(pmids) > 0

@pytest.mark.skip()
def test_query_title_and_abstract():
    pmid = "38987616"
    title, abstract, is_preprint = query_title_abstract_ispreprint(pmid)
    logger.info(title)
    logger.info(abstract)
    assert title is not None
    assert abstract is not None

@pytest.mark.skip()
def test_full_text():
    pmid = "40462993" # "40451396" # "16143486" # "38987616"
    res, html_content = query_full_text(pmid)
    assert res
    assert html_content is not None

    extractor = HtmlTableExtractor()
    title = extractor.extract_title(html_content)
    assert title is not None and len(title) > 0
    tables = extractor.extract_tables(html_content)
    assert tables is not None
    assert len(tables) > 0

    sections = extractor.extract_sections(html_content)
    assert sections is not None
   
    full_text = "\n".join([sec["section"] + "\n" + sec["content"] for sec in sections])
    assert full_text is not None and len(full_text) > 0
    
    data_availability = extractor.extract_data_availability(html_content)
    assert data_availability is not None and len(data_availability) > 0
    

@pytest.mark.skip()
def test_query_statement_with_query_count():
    query = '(Alzheimer AND ("single cell" OR "single nucleus" OR "single-cell")) AND ("RNA sequencing" OR "RNA-seq" OR "single-cell RNA-seq")'
    cnt = query_count(
        query,
        mindate="2024/06/01",
        maxdate="2025/06/01",
        datetype="pdat",
    )
    print(cnt)
    logger.info(f'{query} count: {cnt}')

@pytest.mark.skip()
def test_convert_html_to_plaintext_for_preprint():
    pmids_preprint = ["40462993",]
    for pmid in pmids_preprint:
        res, html_content = query_full_text(pmid)
        assert res
        assert html_content is not None

        plaintext = convert_html_to_plaintext(html_content)
        assert plaintext is not None and len(plaintext) > 0
        assert "Data Availability" not in plaintext and "Methods" not in plaintext, "preprint should not have these sections"

@pytest.mark.skip()
def test_convert_html_to_plaintext():
    # curated list of PMIDs to test
    pmids = [
    "37930842", "38167548", "38181047", "38263132", "38265551", "38266073",
    "38331937", "38464521", "38491918", "38499592", "38514782", "38514804",
    "38521932", "38531951", "38542263", "38554701", "38557897", "38619646",
    "38626772", "38821054", "38844475", "38846640", "38923164", "38924962",
    "38976461", "38987616", "39062541", "39068182", "39198642", "39299805",
    "39406950", "39438735", "39543008", "39591421", "39733087", "39754611",
    "39814561", "39936500", "39945772", "39970860", "39993589", "40021385",
    "40026671", "40037709", "40071147", "40122810", "40168986", "40267277",
    "40307569", "40442087", "40448997", "40470298"
]
    # Qi's list of PMIDs to test
    # ["40060041", "40093585", "40112813", "40166175", "40196633", "40205047",
    #          "40264187", "40275379", "40329537", "40336141", "40346662", "40381615",
    #          "40537665",]
    # ["39537608", "39554174", "39580485", "39641382", "39928878", "40050704",]
    # ["39150395"] # ["39147742", "39150395", "39265576", "39308178", "39500314", "39528672",]
    # ["38695952", "38826305", "38867189", "38902234", "38918830", "39048816",]
    # ["38417436", "38453894", "38472200", "38480892", "38566045", "38600587",]
    # ["37645844", "38228117", "38340719", "38347225", "38352874", "38385967", ]
    # ["37930842", "38167548", "38181047", "38263132", "38265551", "38266073"]
    for pmid in pmids:
        res, html_content = query_full_text(pmid)
        assert res
        assert html_content is not None

        plaintext = convert_html_to_plaintext(html_content)
        assert plaintext is not None and len(plaintext) > 0
        if not "Data Availability" in plaintext:
            logger.warning(f"PMID {pmid}: should have Data Availability section")
        if not "Methods" in plaintext:
            logger.warning(f"PMID {pmid}: should have Methods section")

def test_html_extractor_for_methods():
    pmids = ["38181047", "38846640", "39068182"] # Example PMID
    for pmid in pmids:
        res, html_content = query_full_text(pmid)
        assert res
        assert html_content is not None

        extractor = HtmlTableExtractor()

        methods = extractor.extract_methods(html_content)
        assert methods is not None and len(methods) > 0
        
    