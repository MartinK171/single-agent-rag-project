import pytest
from unittest.mock import Mock, patch
from src.integration.rag_pipeline import RAGPipeline
from src.router.types import QueryType
import json

class MockVectorStore:
    """Mock Vector Store for testing."""
    def __init__(self):
        self.search_results = []

    def search(self, query: str, limit: int = 3):
        """Simulate search with mock results."""
        return self.search_results

    def set_results(self, results):
        """Set mock search results."""
        self.search_results = results

class MockResult:
    """Mock search result."""
    def __init__(self, text: str):
        self.text = text

@pytest.fixture
def sample_context():
    return "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval with LLMs."

@pytest.fixture
def mock_vector_store():
    return MockVectorStore()

@pytest.fixture
def mock_llm():
    mock = Mock()
    def invoke_side_effect(prompt, **kwargs):
        if 'Expand this query to improve search results:' in prompt:
            return 'what is rag?'
        elif prompt.startswith('Please calculate:'):
            return "The result is 50"
        elif 'Use the following context to help answer the question.' in prompt:
            return "Here is the answer using the context."
        else:
            # Return appropriate JSON for the router
            return json.dumps({
                "query_type": "RETRIEVAL",
                "confidence": 0.9,
                "should_retrieve": True,
                "retrieval_query": "Expanded retrieval query",
                "reasoning": "Mock reasoning for routing."
            })
    mock.invoke.side_effect = invoke_side_effect
    return mock

@pytest.fixture
def rag_pipeline(mock_llm, mock_vector_store):
    with patch('src.integration.rag_pipeline.OllamaLLM', return_value=mock_llm):
        return RAGPipeline(vector_store=mock_vector_store)

def test_query_transformation(rag_pipeline, mock_llm):
    """Test query transformation."""
    query = "  What is RAG?  "
    transformed = rag_pipeline._transform_query(query, QueryType.RETRIEVAL)
    assert transformed.strip() == "what is rag?"

def test_retrieval_with_context(rag_pipeline, mock_vector_store, sample_context):
    """Test retrieval with context."""
    # Set up mock results
    mock_vector_store.set_results([MockResult(sample_context)])
    
    result = rag_pipeline.process_query("What is RAG?")
    
    assert result["success"] is True
    assert "response" in result
    assert result["query_type"] == "retrieval"
    assert sample_context in result.get("context_used", "")

def test_calculation_handling(rag_pipeline, mock_llm):
    """Test calculation query handling."""
    def invoke_side_effect(prompt, **kwargs):
        if prompt.startswith('Please calculate:'):
            return "The result is 50"
        else:
            # Return appropriate JSON for the router
            return json.dumps({
                "query_type": "CALCULATION",
                "confidence": 0.9,
                "should_retrieve": False,
                "retrieval_query": "",
                "reasoning": "Mock reasoning for calculation."
            })
    mock_llm.invoke.side_effect = invoke_side_effect

    result = rag_pipeline.process_query("What is 25% of 200?")

    assert "response" in result
    assert result.get("success") is True
    assert result["response"] == "The result is 50"

def test_clarification_handling(rag_pipeline, mock_llm):
    """Test clarification query handling."""
    def invoke_side_effect(prompt, **kwargs):
        if 'Use the following context to help answer the question.' in prompt:
            return "I need more information."
        else:
            return json.dumps({
                "query_type": "CLARIFICATION",
                "confidence": 0.8,
                "should_retrieve": False,
                "retrieval_query": "",
                "reasoning": "Mock reasoning for clarification."
            })
    mock_llm.invoke.side_effect = invoke_side_effect

    result = rag_pipeline.process_query("How does it work?")

    assert result.get("clarification_needed") is True
    assert "more information" in result.get("response", "").lower()
    assert result.get("success") is True

def test_error_handling(rag_pipeline, mock_llm):
    """Test error handling."""
    # Simulate an exception when the LLM is invoked
    mock_llm.invoke.side_effect = Exception("Test error")
    
    result = rag_pipeline.process_query("test query")
    
    assert result["success"] is False
    assert "error" in result
    assert result["error"] == "Test error"
    assert "Failed to process direct query" in result["response"]


def test_retry_mechanism(rag_pipeline):
    """Test retry mechanism."""
    failing_function = Mock(side_effect=[Exception("Error"), Exception("Error"), "Success"])
    
    result = rag_pipeline._handle_with_retry(failing_function, "test query")
    
    assert result == "Success"

def test_context_augmentation(rag_pipeline):
    """Test context augmentation."""
    query = "What is RAG?"
    context = "RAG is a technique for enhancing LLM responses."
    
    augmented = rag_pipeline._augment_with_context(query, context)
    
    assert query in augmented
    assert context in augmented
    assert "context" in augmented.lower()

def test_fallback_mechanism(rag_pipeline, mock_vector_store):
    """Test fallback when retrieval fails."""
    # Ensure vector store returns no results
    mock_vector_store.set_results([])
    
    result = rag_pipeline.process_query("What is RAG?")
    
    assert result["success"] is False
    assert "No relevant information" in result.get("response", "")

