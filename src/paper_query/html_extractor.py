import re

from bs4 import BeautifulSoup, FeatureNotFound, Tag
from typing import Callable, Optional
from typing import List, Optional, Dict

import pandas as pd
from .table_utils import html_table_to_markdown, dataframe_to_markdown
from .utils import convert_html_table_to_dataframe, escape_braces_for_format


def get_tag_text(tag: Tag) -> str:
    text = tag.text
    text = text.strip()
    if text is not None and len(text) > 0:
        return text
    children = list(tag.descendants)
    if len(children) == 0:
        return text
    for child in children:
        the_text = child.text
        text += the_text
    return text

def extract_data_availability(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
        
    # Find a heading tag that contains the word "data availability" (case-insensitive)
    data_availability_heading = soup.find(
        lambda tag: tag.name in ["h1", "h2", "h3"] and "data" in tag.get_text(strip=True).lower() and "availability" in tag.get_text(strip=True).lower()
    )
    
    if not data_availability_heading:
        return None
    # Find the parent <section> or container that wraps the data availability section
    data_availability_section = data_availability_heading.find_parent("section")
    if not data_availability_section:
        data_availability_section = data_availability_heading.find_parent("div")
        if not data_availability_section:
            # return None # No wrapping section found 
            return None
    
    text = ""
    for child in data_availability_section.children:
        if child.name == "h1" or child.name == "h2" or child.name == "h3":
            # Skip headings within the data availability section
            continue
        child_text = child.get_text(separator=" ", strip=True)
        if child_text:
            text += child_text + "\n"
    return text.strip() if text else None

def extract_methods(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    
    # Find a heading tag that contains the word "methods" (case-insensitive)
    method_names = ["methods", "methodology", "method", "mothodologies"]
    methods_heading = soup.find(
        lambda tag: tag.name in ["h1", "h2", "h3"] and any(x in tag.get_text(strip=True).lower() for x in method_names)
    )

    
    if not methods_heading:
        return None  # No methods heading found
    
    # Find the parent <section> or container that wraps the methods section
    methods_section = methods_heading.find_parent("section")
    if not methods_section:
        methods_section = methods_heading.find_parent("div")
        if not methods_section:
            # return None # No wrapping section found 
            return None
    
    text = ""
    for child in methods_section.children:
        if child.name == "h2" or child.name == "h3":
            # Skip headings within the methods section
            continue
        child_text = child.get_text(separator=" ", strip=True)
        if child_text:
            text += child_text + "\n"
    
    return text.strip() if text else None

class HtmlTableParser(object):
    MAX_LEVEL = 3
    CAPTION_TAG_CANDIDATES = ["figcaption"]
    CAPTION_CANDIDATES = ["caption", "captions", "title"]
    FOOTNOTE_CANDIDATES = ["note", "legend", "description", "foot", "notes"]

    def __init__(self):
        pass

    def _get_caption_or_footnote_text(self, tag: Tag) -> str:
        return get_tag_text(tag)

    @staticmethod
    def _is_caption_in_text(text):
        for cap in HtmlTableParser.CAPTION_CANDIDATES:
            if cap in text:
                return True
        return False
    
    @staticmethod
    def _is_caption_by_tagname(tagname: str) -> bool:
        return tagname in HtmlTableParser.CAPTION_TAG_CANDIDATES

    @staticmethod
    def _is_footnote_in_text(text):
        for foot in HtmlTableParser.FOOTNOTE_CANDIDATES:
            if foot in text:
                return True
        return False

    def _find_caption_and_footnote_recursively(
        self,
        parent_tag: Tag | None,
        level: int,
        found_caption: Optional[bool] = False,
        found_footnote: Optional[bool] = False,
    ) -> tuple[str | None, str | None, Tag | None]:
        if parent_tag is None:
            return "", "", None
        if level > HtmlTableParser.MAX_LEVEL:
            return "", "", None
        children = parent_tag.children
        caption = None
        footnote = None
        for child in children:
            if not hasattr(child, "attrs"):
                continue
            if hasattr(child, "name") and HtmlTableParser._is_caption_by_tagname(tagname=child.name):
                caption = self._get_caption_or_footnote_text(child)
                found_caption = True
                continue
            classes = child.attrs.get("class")
            if classes is None:
                continue
            if not isinstance(classes, str):
                try:
                    classes = " ".join(classes)
                except:
                    continue
            if not found_caption and HtmlTableParser._is_caption_in_text(classes):
                caption = self._get_caption_or_footnote_text(child)
                found_caption = True
            if not found_footnote and HtmlTableParser._is_footnote_in_text(classes):
                footnote = self._get_caption_or_footnote_text(child)
                found_footnote = True
        if found_caption and found_footnote:
            return caption, footnote, parent_tag
        if not found_caption and not found_footnote:
            return self._find_caption_and_footnote_recursively(
                parent_tag.parent, level + 1, found_caption, found_footnote
            )
        if not found_caption:
            caption, _, further_parent_tag = (
                self._find_caption_and_footnote_recursively(
                    parent_tag.parent, level + 1, found_caption, found_footnote
                )
            )
            final_parent_tag = (
                further_parent_tag
                if further_parent_tag is not None
                and (caption is not None and len(caption) > 0)
                else parent_tag
            )
            return caption, footnote, final_parent_tag
        if not found_footnote:
            _, footnote, further_parent_tag = (
                self._find_caption_and_footnote_recursively(
                    parent_tag.parent, level + 1, found_caption, found_footnote
                )
            )
            final_parent_tag = (
                further_parent_tag
                if further_parent_tag is not None
                and (footnote is not None and len(footnote) > 0)
                else parent_tag
            )
            return caption, footnote, final_parent_tag
        
        return None, None, None

    def _find_caption_and_footnote(self, table_tag: Tag):
        return self._find_caption_and_footnote_recursively(table_tag.parent, 1)

    def extract_tables(self, html: str):
        soup = BeautifulSoup(html, "html.parser")
        tags = soup.select("table")
        tables = []
        for tag in tags:
            strTag = str(tag)
            table = convert_html_table_to_dataframe(strTag)
            if table is None:
                continue
            caption, footnote, parent_tag = self._find_caption_and_footnote(tag)
            parent_tag = parent_tag if parent_tag is not None else tag
            tables.append(
                {
                    "caption": caption if caption is not None else "",
                    "footnote": footnote if footnote is not None else "",
                    "table": table,
                    "raw_tag": str(parent_tag),
                }
            )
        return tables
    
    def _traverse_up(self, cur: Tag | None, level: int, max_level: int, check_cb: Callable):
        if cur is None:
            return False
        if level == max_level:
            return False
        res = check_cb(cur)
        return res if res else self._traverse_up(cur.parent, level+1, max_level, check_cb)
    
    def _traverse_down(self, cur: Tag | None, level: int, max_level: int, check_cb: Callable):
        if cur is None:
            return False
        if level == max_level:
            return False
        res = check_cb(cur)
        if res:
            return res
        try:
            for child in cur.children:
                res = self._traverse_down(child, level+1, max_level, check_cb)
                if res:
                    return res
        except AttributeError:
            return False
        return False

    def extract_title(self, html: str):
        def check_title_in_tag_classes(tag: Tag):
            if tag is None:
                return False
            try:
                if tag.attrs is None:
                    return False
            except AttributeError:
                return False
            classes = tag.attrs.get("class")
            if classes is None:
                return False
            if not isinstance(classes, str):
                try:
                    classes = " ".join(classes)
                except:
                    return False
            title_in_classes = "title" in classes
            if title_in_classes:
                return True
            id = tag.attrs.get('id')
            if id is not None and "title" in id.lower():
                return True
            else:
                return False
        
        soup = BeautifulSoup(html, "html.parser")
        tags = soup.select("h1")
        for tag in tags:
            if self._traverse_up(tag, 1, 5, check_title_in_tag_classes):
                return get_tag_text(tag)
            if self._traverse_down(tag, 1, 3, check_title_in_tag_classes):
                return get_tag_text(tag)
        
        return None

    def _find_first_occurrence(self,
                               soup: BeautifulSoup,
                               keywords: List[str]) -> Optional[Tag]:
        kw_lower = [k.lower() for k in keywords]
        valid_tags = {"section", "div", "article", "main", "h1", "h2", "h3", "p", "ul", "ol", "table", "article"}

        for el in soup.find_all(True):
            if el.name not in valid_tags:
                continue
            if any(kw in cls.lower() for cls in el.get("class", []) for kw in kw_lower):
                return el
            if any(kw in (el.get("id", "").lower()) for kw in kw_lower):
                return el
            direct = ''.join(el.find_all(string=True, recursive=False)).strip().lower()
            if any(kw in direct for kw in kw_lower):
                return el
        return None

    def extract_abstract(self, html: str):
        sections = self.extract_sections(html)
        if sections == None:
            return None
        for section in sections:
            if "abstract" in section["section"].lower():
                return section["content"].replace("\n", " ")
        return (sections[0]["section"] + "\n" + sections[0]["content"]).replace("\n", " ") + "\n......" or None
    
    def extract_sections(self, html: str):
        """
        Generic section extraction for main body content (non-PMC).
        Returns [{'section': ..., 'content': ...}, ...]

        References and acknowledgements are skipped (not included), but
        traversal continues past them so later sections — supplementary
        material, data availability, etc. — are still captured.
        """
        skip_sections = [
                "reference", "references",
                # "acknowledgement", "acknowledgment",
                # "acknowledgements", "acknowledgments",
        ]

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a"):
            a.decompose()

        start = self._find_first_occurrence(soup, ["abstract"])
        if not start:
            return None

        sections: List[Dict[str, str]] = []
        current: Optional[Dict[str, str]] = None
        seen_global = set()
        heading_tags = ["h1", "h2", "h3", "h4"]
        block_tags = ["p", "ul", "ol", "div", "section", "article"]

        for el in start.find_all_next():
            # ── 1. Handle section headings ───────────────────────────────
            if el.name in heading_tags:
                h_raw = el.get_text(strip=True)
                h_low = h_raw.lower()

                # Finalize previous section before starting a new one
                if current and current["content"].strip():
                    lines = list(dict.fromkeys(current["content"].splitlines()))
                    current["content"] = "\n".join(lines).strip()
                    sections.append(current)

                # References / acknowledgements: skip body, keep traversing
                if any(kw in h_low for kw in skip_sections):
                    current = None
                    continue

                current = {"section": h_raw, "content": ""}
                continue

            # ── 2. Collect main body text ───────────────────────────────
            if current is None:
                continue  # Not yet in the first main body section

            if el.name == "table":
                try:
                    df = convert_html_table_to_dataframe(str(el))
                    # I use this one instead of the custom html_table_to_dataframe implementation,
                    # because it uses StringIO and is likely more robust.
                    # That said, the previous custom version hasn’t caused any major issues so far,
                    # so I’m not eager to change it either. 
                    markdown = dataframe_to_markdown(df)
                    if markdown and markdown not in seen_global:
                        current["content"] += markdown + "\n"
                        seen_global.add(markdown)
                    continue
                except Exception:
                    pass

            if el.name in block_tags:
                txt = el.get_text(separator="\n", strip=True)
                if txt and txt not in seen_global:
                    current["content"] += txt + "\n"
                    seen_global.add(txt)

        # Document ended but still has current section
        if current and current["content"].strip():
            lines = list(dict.fromkeys(current["content"].splitlines()))
            current["content"] = "\n".join(lines).strip()
            sections.append(current)

        return sections


class PMCHtmlTableParser(object):
    def __init__(self):
        pass

    def extract_tables(self, html: str):
        soup = BeautifulSoup(html, "html.parser")
        tags = soup.select("div.table-wrap.anchored.whole_rhythm")
        tables = []
        for tag in tags:
            tbl_soup = BeautifulSoup(str(tag), "html.parser")
            caption = tbl_soup.select("div.caption")
            caption = caption[0].text if len(caption) > 0 else ""
            table = tbl_soup.select("div.xtable")
            table = str(table[0]) if len(table) > 0 else ""
            table = convert_html_table_to_dataframe(table)
            footnote = tbl_soup.select("div.tblwrap-foot")
            footnote = footnote[0].text if len(footnote) > 0 else ""
            tables.append(
                {
                    "caption": caption,
                    "table": table,
                    "footnote": footnote,
                    "raw_tag": str(tag),
                }
            )

        return tables

    def extract_title(self, html: str):
        soup = BeautifulSoup(html, "html.parser")
        tags = soup.select("hgroup h1")
        for tag in tags:
            text = get_tag_text(tag)
            if len(text.strip()) > 0:
                return text.strip()
        return None

    def extract_abstract(self, html: str):
        """
        """
        soup = BeautifulSoup(html, "html.parser")

        # Find a heading tag that contains the word "abstract" (case-insensitive)
        abstract_heading = soup.find(
            lambda tag: tag.name in ["h2", "h3"] and "abstract" in tag.get_text(strip=True).lower()
        )

        if not abstract_heading:
            return None  # No abstract heading found

        # Find the parent <section> or container that wraps the abstract
        abstract_section = abstract_heading.find_parent("section")
        if not abstract_section:
            return None  # No wrapping section found

        # Extract all <p> tags (paragraphs) under the abstract section
        abstract_paragraphs = abstract_section.find_all("p")

        # Combine the text from each paragraph
        abstract_text = "\n".join(p.get_text(separator=" ", strip=True) for p in abstract_paragraphs)

        return abstract_text.strip()
    
    def extract_sections(self, html: str):
        """
        Extracts sections (h2/h3) and content starting from 'Abstract'.

        References and acknowledgements are skipped (not included), but
        traversal continues past them so later sections — supplementary
        material, data availability, etc. — are still captured.
        """
        skip_sections = [
            "reference", "references",
            # "acknowledgement", "acknowledgment",
            # "acknowledgements", "acknowledgments",
        ]
        soup = BeautifulSoup(html, "html.parser")
        body = soup.body
        if not body:
            return []

        heading_tags = ["h2", "h3"]
        sections = []
        current_section = None
        started = False

        for element in body.descendants:
            if isinstance(element, Tag):
                if element.name in heading_tags:
                    heading_text = element.get_text(strip=True).lower()

                    if not started and "abstract" in heading_text:
                        started = True
                        current_section = {
                            "section": element.get_text(strip=True),
                            "content": ""
                        }
                        continue

                    if started:
                        # Finalize previous section before starting a new one
                        if current_section:
                            current_section["content"] = current_section["content"].strip()
                            sections.append(current_section)
                            current_section = None

                        # References / acknowledgements: skip body, keep traversing
                        if any(x in heading_text for x in skip_sections):
                            continue

                        current_section = {
                            "section": element.get_text(strip=True),
                            "content": ""
                        }

                elif started and current_section:
                    # Tables: convert HTML
                    if element.name == "table" or (
                    "xtable" in element.get("class", []) if element.has_attr("class") else False):
                        current_section["content"] += html_table_to_markdown(str(element)) + "\n"

                    # Text elements: convert to plain text
                    elif element.name in ["p", "ul", "ol"]:
                        text = element.get_text(separator=" ", strip=True)
                        if text:
                            current_section["content"] += text + "\n"

        if current_section:
            current_section["content"] = current_section["content"].strip()
            sections.append(current_section)

        return sections


class XmlTableParser:
    """Parser for JATS/NLM XML returned by NCBI efetch (db=pmc, retmode=xml).

    References are skipped (excluded from output) but traversal continues —
    sections after references (supplementary material, acknowledgements, etc.)
    are still captured.
    """

    SKIP_SECTION_KEYWORDS = ["reference", "references"]
    TEXT_BLOCK_TAGS = {"p", "list", "boxed-text", "disp-quote"}

    @staticmethod
    def _is_xml_content(content: str) -> bool:
        if not content:
            return False
        lowered = content.lstrip().lower()
        if lowered.startswith("<?xml"):
            return True
        if "<!doctype article" in lowered:
            return True
        markers = ["<article-meta", "<article-title", "<table-wrap", "<sec ", "<sec>", "<abstract>", "<abstract "]
        return any(m in lowered for m in markers)

    def _parse_xml(self, content: str) -> Optional[BeautifulSoup]:
        if not self._is_xml_content(content):
            return None
        try:
            return BeautifulSoup(content, "xml")
        except FeatureNotFound:
            return BeautifulSoup(content, "html.parser")

    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""
        return re.sub(r"\s+", " ", text).strip()

    def _extract_abstract_text(self, soup: BeautifulSoup) -> Optional[str]:
        article_meta = soup.find("article-meta")
        abstract = article_meta.find("abstract") if article_meta else soup.find("abstract")
        if abstract is None:
            return None

        text_parts = []
        sections = abstract.find_all("sec", recursive=False)
        if sections:
            for sec in sections:
                title_tag = sec.find("title", recursive=False)
                sec_title = self._clean_text(title_tag.get_text(" ", strip=True)) if title_tag else ""
                paras = [
                    self._clean_text(p.get_text(" ", strip=True))
                    for p in sec.find_all("p")
                    if self._clean_text(p.get_text(" ", strip=True))
                ]
                sec_text = " ".join(paras)
                if sec_title and sec_text:
                    text_parts.append(f"{sec_title}: {sec_text}")
                elif sec_text:
                    text_parts.append(sec_text)
        else:
            paras = [
                self._clean_text(p.get_text(" ", strip=True))
                for p in abstract.find_all("p")
                if self._clean_text(p.get_text(" ", strip=True))
            ]
            if paras:
                text_parts.extend(paras)
            else:
                fallback = self._clean_text(abstract.get_text(" ", strip=True))
                if fallback:
                    text_parts.append(fallback)

        res = "\n".join(text_parts).strip()
        return res if res else None

    def _extract_table_caption(self, table_wrap: Tag) -> str:
        label_tag = table_wrap.find("label", recursive=False)
        caption_tag = table_wrap.find("caption", recursive=False) or table_wrap.find("caption")
        label = self._clean_text(label_tag.get_text(" ", strip=True)) if label_tag else ""
        caption = self._clean_text(caption_tag.get_text(" ", strip=True)) if caption_tag else ""
        if label and caption:
            return caption if caption.lower().startswith(label.lower()) else f"{label} {caption}".strip()
        return label or caption

    def _extract_table_footnote(self, table_wrap: Tag) -> str:
        foot = table_wrap.find("table-wrap-foot")
        if foot is None:
            return ""
        parts, seen = [], set()
        fn_tags = foot.find_all("fn")
        if fn_tags:
            for fn in fn_tags:
                text = self._clean_text(fn.get_text(" ", strip=True))
                if text and text not in seen:
                    seen.add(text)
                    parts.append(text)
        else:
            text = self._clean_text(foot.get_text(" ", strip=True))
            if text:
                parts.append(text)
        return "\n".join(parts)

    def extract_tables(self, content: str):
        soup = self._parse_xml(content)
        if soup is None:
            return []
        tables = []
        for tw in soup.find_all("table-wrap"):
            table_tag = tw.find("table")
            if table_tag is None:
                continue
            df = convert_html_table_to_dataframe(str(table_tag))
            if df is None:
                continue
            tables.append({
                "caption": self._extract_table_caption(tw),
                "footnote": self._extract_table_footnote(tw),
                "table": df,
                "raw_tag": str(tw),
            })
        if tables:
            return tables
        for table_tag in soup.find_all("table"):
            df = convert_html_table_to_dataframe(str(table_tag))
            if df is None:
                continue
            tables.append({"caption": "", "footnote": "", "table": df, "raw_tag": str(table_tag)})
        return tables

    def extract_title(self, content: str) -> Optional[str]:
        soup = self._parse_xml(content)
        if soup is None:
            return None
        article_meta = soup.find("article-meta")
        if article_meta is not None:
            title_tag = article_meta.find("article-title")
            if title_tag is not None:
                title = self._clean_text(title_tag.get_text(" ", strip=True))
                if title:
                    return title
        for title_tag in soup.find_all("article-title"):
            if title_tag.find_parent("ref-list") or title_tag.find_parent("citation"):
                continue
            title = self._clean_text(title_tag.get_text(" ", strip=True))
            if title:
                return title
        return None

    def extract_abstract(self, content: str) -> Optional[str]:
        soup = self._parse_xml(content)
        if soup is None:
            return None
        return self._extract_abstract_text(soup)

    def _section_is_skipped(self, title: str, sec_type: str) -> bool:
        key = f"{title} {sec_type}".lower()
        return any(kw in key for kw in self.SKIP_SECTION_KEYWORDS)

    def _extract_section_content(self, section: Tag) -> str:
        parts, seen = [], set()
        for child in section.children:
            if not isinstance(child, Tag):
                continue
            if child.name in {"title", "sec"}:
                continue
            if child.name == "table-wrap":
                table_tag = child.find("table")
                if table_tag is None:
                    continue
                df = convert_html_table_to_dataframe(str(table_tag))
                if df is None:
                    continue
                md = dataframe_to_markdown(df).strip()
                if md and md not in seen:
                    seen.add(md)
                    parts.append(md)
                continue
            if child.name in self.TEXT_BLOCK_TAGS:
                text = self._clean_text(child.get_text(" ", strip=True))
                if text and text not in seen:
                    seen.add(text)
                    parts.append(text)
                continue
            text = self._clean_text(child.get_text(" ", strip=True))
            if text and text not in seen:
                seen.add(text)
                parts.append(text)
        return "\n".join(parts).strip()

    def _extract_sections_from_sec(self, sec: Tag) -> List[Dict[str, str]]:
        sections = []
        title_tag = sec.find("title", recursive=False)
        section_title = self._clean_text(title_tag.get_text(" ", strip=True)) if title_tag else ""
        sec_type = self._clean_text(sec.attrs.get("sec-type", ""))

        # Skip this section's content but return empty so the caller continues to siblings
        if self._section_is_skipped(section_title, sec_type):
            return sections

        if not section_title:
            section_title = sec_type or self._clean_text(sec.attrs.get("id", ""))

        content = self._extract_section_content(sec)
        if section_title and content:
            sections.append({"section": section_title, "content": content})

        for child_sec in sec.find_all("sec", recursive=False):
            sections.extend(self._extract_sections_from_sec(child_sec))

        return sections

    def extract_sections(self, content: str) -> Optional[List[Dict[str, str]]]:
        soup = self._parse_xml(content)
        if soup is None:
            return None

        sections: List[Dict[str, str]] = []
        abstract_text = self._extract_abstract_text(soup)
        if abstract_text:
            sections.append({"section": "Abstract", "content": abstract_text})

        body_tag = soup.find("body")
        if body_tag is not None:
            for sec in body_tag.find_all("sec", recursive=False):
                sections.extend(self._extract_sections_from_sec(sec))

        # Floating table-wraps outside <body> (e.g. in <floats-group>)
        all_tws = soup.find_all("table-wrap")
        body_tws = set(body_tag.find_all("table-wrap")) if body_tag else set()
        floating = [t for t in all_tws if t not in body_tws]
        table_parts, seen_tables = [], set()
        for tw in floating:
            table_tag = tw.find("table")
            if table_tag is None:
                continue
            df = convert_html_table_to_dataframe(str(table_tag))
            if df is None:
                continue
            caption = self._extract_table_caption(tw)
            md = dataframe_to_markdown(df).strip()
            if not md or md in seen_tables:
                continue
            seen_tables.add(md)
            table_parts.append(f"{caption}\n{md}".strip() if caption else md)
        if table_parts:
            sections.append({"section": "Tables", "content": "\n\n".join(table_parts)})

        return sections if sections else None


class HtmlTableExtractor(object):
    def __init__(self):
        self.xml_parser = XmlTableParser()
        self.html_parsers = [
            PMCHtmlTableParser(),
            HtmlTableParser(),
        ]

    def _get_parsers(self, content: str):
        if XmlTableParser._is_xml_content(content):
            return [self.xml_parser]
        return self.html_parsers

    def extract_tables(self, html: str):
        tables = []
        for parser in self._get_parsers(html):
            tables = parser.extract_tables(html)
            if tables and len(tables) > 0:
                break

        tables = HtmlTableExtractor._remove_duplicate(tables)
        return tables

    def extract_title(self, html: str):
        for parser in self._get_parsers(html):
            title = parser.extract_title(html)
            if title is not None:
                return escape_braces_for_format(title)

        return None

    def extract_abstract(self, html: str):
        for parser in self._get_parsers(html):
            abstract = parser.extract_abstract(html)
            if abstract is not None:
                return escape_braces_for_format(abstract)

        return None

    def extract_data_availability(self, html: str):
        return extract_data_availability(html)

    def extract_methods(self, html: str):
        return extract_methods(html)

    def extract_sections(self, html: str) -> dict | None:
        for parser in self._get_parsers(html):
            sections = parser.extract_sections(html)
            if sections is not None:
                for s in sections:
                    if "section" in s:
                        s["section"] = escape_braces_for_format(s["section"])
                    if "content" in s:
                        s["content"] = escape_braces_for_format(s["content"])
                return sections
        return None

    @staticmethod
    def _tables_eq(tablel: dict, table2: dict) -> bool:
        df_table1: pd.DataFrame = tablel["table"]
        df_table2: pd.DataFrame = table2["table"]

        return df_table1.equals(df_table2)

    @staticmethod
    def _remove_duplicate(tables):
        res_tables = []
        for table in tables:
            if len(res_tables) == 0:
                res_tables.append(table)
                continue
            prev_table = res_tables[-1]
            if HtmlTableExtractor._tables_eq(prev_table, table):
                continue
            res_tables.append(table)

        return res_tables
