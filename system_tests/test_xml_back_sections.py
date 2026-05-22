"""
Offline tests for XmlTableParser back-matter extraction.

Verifies that extract_sections() now captures the JATS <back> sections —
specifically ACKNOWLEDGEMENTS (<ack>) and DATA AVAILABILITY (<fn-group>/<fn>) —
while still skipping the reference list. Uses the checked-in fixture
tests/data/33432193.xml; no network or LLM calls.
"""

import os

import pytest

from src.paper_query.html_extractor import HtmlTableExtractor, XmlTableParser

XML_PATH = os.path.join(
    os.path.dirname(__file__), "..", "tests", "data", "33432193.xml"
)


@pytest.fixture(scope="module")
def xml_content() -> str:
    with open(XML_PATH, encoding="utf-8") as f:
        return f.read()


def _section(sections, keyword):
    return next(
        (s for s in sections if keyword.lower() in s["section"].lower()), None
    )


def test_extract_sections_includes_acknowledgements(xml_content):
    sections = XmlTableParser().extract_sections(xml_content)
    assert sections is not None

    ack = _section(sections, "acknowledgement")
    assert ack is not None, [s["section"] for s in sections]
    assert "Kampmann" in ack["content"]


def test_extract_sections_includes_data_availability(xml_content):
    sections = XmlTableParser().extract_sections(xml_content)
    assert sections is not None

    da = _section(sections, "data availability")
    assert da is not None, [s["section"] for s in sections]
    # The statement names GEO + Synapse accessions.
    assert "GSE147528" in da["content"]
    assert "syn21788402" in da["content"]
    # The heading line itself must not be duplicated into the body.
    assert not da["content"].lower().startswith("data availability")


def test_other_fn_group_statements_become_sections(xml_content):
    sections = XmlTableParser().extract_sections(xml_content)
    titles_upper = {s["section"].upper() for s in sections}
    # Sibling <fn> footnotes are each their own section.
    assert "CODE AVAILABILITY" in titles_upper
    assert "ACCESSION CODES" in titles_upper


def test_references_are_skipped(xml_content):
    sections = XmlTableParser().extract_sections(xml_content)
    assert not any("reference" in s["section"].lower() for s in sections)


def test_facade_routes_xml_and_keeps_back_sections(xml_content):
    sections = HtmlTableExtractor().extract_sections(xml_content)
    assert sections is not None
    joined = " ".join(s["section"] for s in sections).upper()
    assert "ACKNOWLEDGEMENTS" in joined
    assert "DATA AVAILABILITY" in joined
