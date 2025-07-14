
import logging

from ..paper_query.html_extractor import HtmlTableExtractor

logger = logging.getLogger(__name__)

def convert_html_to_plaintext(
    html: str, 
    include_data_availability=True,
    include_methods=True,
) -> str | None:
    """
    Converts HTML content to plain text by removing HTML tags and decoding HTML entities.
    Args:
        html (str): The HTML content to convert.
    Returns:
        str: The plain text content.
    """
    try:
        extractor = HtmlTableExtractor()
        sections = extractor.extract_sections(html)
        if sections is None:
            return None
        if include_data_availability:
            data_availability = extractor.extract_data_availability(html)
            if data_availability:
                sections.append({"section": "Data Availability", "content": data_availability})
        
        if include_methods:
            methods = extractor.extract_methods(html)
            if methods:
                sections.append({"section": "Methods", "content": methods})
        return "\n".join([sec["section"].strip() + "\n" + sec["content"].strip() \
                          for sec in sections])
    except Exception as e:
        logger.error(f"Error converting HTML to plaintext: {e}")
        return None  # Return original HTML if conversion fails
    
def obtain_full_text(pmid: str) -> str | None:
    """
    Obtain the full text of a paper given its PubMed ID (PMID).
    Args:
        pmid (str): The PubMed ID of the paper.
    Returns:
        str: The full text content of the paper, or None if not available.
    """
    try:
        from ..paper_query.pubmed_query import query_full_text
        res, html = query_full_text(pmid)
        if not res or html is None:
            return None
        return convert_html_to_plaintext(html)
    except Exception as e:
        logger.error(f"Error obtaining full text for PMID {pmid}: {e}")
        return None  # Return None if unable to obtain full text