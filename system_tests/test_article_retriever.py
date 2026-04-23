import pytest

from src.paper_query.article_retriever import ArticleRetriever

def test_article_retriever():
    retriever = ArticleRetriever()
    res, content, code = retriever.request_article("33083725")
    assert res
