import pytest
from unittest.mock import Mock, patch
from src.router.chain import RouterChain
from src.router.types import QueryType, RouterResponse
from langchain.prompts import PromptTemplate
import json

class MockOllamaLLM:
    """Mock OllamaLLM class to simulate LLM behavior."""

    def __init__(self, default_response: str = ""):
        self.model = "llama2"
        self.host = "ollama"
        self.port = 11434
        self._response = default_response

    def invoke(self, prompt: str, **kwargs) -> str:
        """Simulate LLM response."""
        return self._response

    def set_response(self, response: str):
        """Set the LLM response for testing."""
        self._response = response

@pytest.fixture
def default_response():
    """Provide a default test response."""
    return json.dumps({
        "query_type": "RETRIEVAL",
        "confidence": 0.9,
        "should_retrieve": True,
        "retrieval_query": "What does the document say about X?",
        "reasoning": "This query requires retrieving information directly from the document."
    })

@pytest.fixture
def mock_llm(default_response):
    """Create a mock LLM with the default response."""
    return MockOllamaLLM(default_response)

@pytest.fixture
def router(mock_llm):
    """Create a RouterChain instance with the mock LLM."""
    with patch('src.router.chain.OllamaLLM', return_value=mock_llm):
        return RouterChain()

def test_router_initialization():
    """Test initialization of RouterChain."""
    with patch('src.router.chain.OllamaLLM') as MockLLM:
        router = RouterChain()
        assert router.llm is not None
        assert router.prompt is not None
        assert isinstance(router.prompt, PromptTemplate)

def test_basic_routing(router, mock_llm):
    """Test basic routing functionality."""
    mock_response = json.dumps({
        "query_type": "RETRIEVAL",
        "confidence": 0.9,  # Updated to match the expected confidence in the assertion
        "should_retrieve": True,
        "retrieval_query": "What does the document say about X?",
        "reasoning": "Test reasoning for retrieval."
    })

    mock_llm.set_response(mock_response)

    result = router.route("What does the document say about X?")
    assert isinstance(result, RouterResponse)
    assert result.query_type == QueryType.RETRIEVAL
    assert result.confidence == 0.9
    assert result.should_retrieve is True
    assert result.metadata["reasoning"] == "Test reasoning for retrieval."

def test_error_handling(router, mock_llm):
    """Test handling of invalid LLM responses."""
    mock_llm.set_response("INVALID RESPONSE FORMAT")

    result = router.route("test query")

    assert isinstance(result, RouterResponse)
    assert result.query_type == QueryType.RETRIEVAL
    assert result.confidence == 0.5  # Expected default confidence in case of error
    assert result.should_retrieve is True
    assert result.retrieval_query == ""
    assert "Failed to parse response." in result.metadata["reasoning"]

@pytest.mark.parametrize("query,expected_type,should_retrieve,confidence", [
    ("What is 2+2?", QueryType.CALCULATION, False, 0.9),
    ("What does the document say?", QueryType.RETRIEVAL, True, 0.8),
    ("What is Python?", QueryType.DIRECT, False, 0.9),
    ("Can you explain more?", QueryType.CLARIFICATION, False, 0.8)
])
def test_query_types(router, mock_llm, query, expected_type, should_retrieve, confidence):
    """Test classification of query types."""
    mock_llm.set_response(json.dumps({
        "query_type": expected_type.value.upper(),
        "confidence": confidence,
        "should_retrieve": should_retrieve,
        "retrieval_query": "",
        "reasoning": "Test reasoning."
    }))

    result = router.route(query)
    assert result.query_type == expected_type
    assert result.should_retrieve == should_retrieve
    assert result.confidence == confidence
    assert "reasoning" in result.metadata

def test_valid_response_parsing(router, mock_llm):
    """Test parsing of a valid JSON response."""
    valid_response = json.dumps({
        "query_type": "DIRECT",
        "confidence": 0.95,
        "should_retrieve": False,
        "retrieval_query": "",
        "reasoning": "This is a simple question."
    })
    mock_llm.set_response(valid_response)

    result = router.route("What is the capital of France?")
    assert result.query_type == QueryType.DIRECT
    assert result.confidence == 0.95
    assert not result.should_retrieve
    assert result.metadata["reasoning"] == "This is a simple question."

def test_fallback_response_on_exception(router, mock_llm):
    """Test fallback response when an exception occurs."""
    # Simulate a scenario where the LLM raises an exception
    with patch.object(mock_llm, 'invoke', side_effect=Exception("LLM failure")):
        result = router.route("Unexpected query")
        assert result.query_type == QueryType.RETRIEVAL
        assert result.confidence == 0.5  # Expected default confidence
        assert result.should_retrieve is True
        assert result.retrieval_query is None
        assert "Error in routing: LLM failure" in result.metadata["reasoning"]

def test_invalid_response_format(router, mock_llm):
    """Test handling of an invalid JSON response."""
    mock_llm.set_response("INVALID JSON")
    result = router.route("test query")

    assert result.query_type == QueryType.RETRIEVAL  # Default fallback
    assert result.confidence == 0.5  # Expected default confidence
    assert result.should_retrieve is True
    assert result.retrieval_query == ""
    assert "Failed to parse response." in result.metadata["reasoning"]
