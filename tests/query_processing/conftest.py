import pytest
from src.query_processing.analyzer import QueryAnalysis

@pytest.fixture
def sample_analysis():
    """Provide a sample QueryAnalysis for tests."""
    return QueryAnalysis(
        complexity=0.5,
        keywords=["test"],
        entities=[],
        topic=None,
        metadata={"has_question_mark": False}
    )

@pytest.fixture
def sample_query():
    """Provide a sample query string."""
    return "What is RAG in machine learning?"