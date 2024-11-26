import pytest
import json
from pathlib import Path
from typing import List, Dict

from src.integration.rag_pipeline import RAGPipeline
from src.router.types import QueryType
from src.vector_db.store import VectorStore
from tests.utils.evaluation import SystemEvaluator
from tests.utils.benchmarks import SystemBenchmark

@pytest.fixture
def test_data_path() -> Path:
    """Get path to test data directory."""
    return Path(__file__).parent.parent.parent / "data" / "test_data"

@pytest.fixture
def test_queries(test_data_path) -> List[Dict]:
    """Load test queries from JSON file."""
    with open(test_data_path / "queries.json") as f:
        data = json.load(f)
        # Flatten queries from different categories
        return (
            data.get("retrieval_queries", []) +
            data.get("direct_queries", []) +
            data.get("calculation_queries", [])
        )

@pytest.fixture
def test_documents(test_data_path) -> List[Dict]:
    """Load test documents from JSON file."""
    with open(test_data_path / "documents.json") as f:
        return json.load(f).get("documents", [])

@pytest.fixture
def vector_store(test_documents) -> VectorStore:
    """Initialize and populate vector store with test documents."""
    store = VectorStore(collection_name="test_collection")
    
    # Add test documents to store
    texts = [doc["content"] for doc in test_documents]
    metadata = [doc["metadata"] for doc in test_documents]
    store.add_texts(texts, metadata)
    
    return store

@pytest.fixture
def rag_system(vector_store) -> RAGPipeline:
    """Initialize RAG system with test configuration."""
    return RAGPipeline(vector_store=vector_store)

def test_system_initialization(rag_system):
    """Test system initializes correctly."""
    assert rag_system is not None
    assert rag_system.vector_store is not None
    assert rag_system.router is not None

def test_retrieval_query(rag_system):
    """Test handling of retrieval queries."""
    query = "What does the documentation say about RAG?"
    result = rag_system.process_query(query)
    
    assert result is not None
    assert result['query_type'] == QueryType.RETRIEVAL.value
    assert result['should_retrieve'] is True
    assert result['confidence'] >= 0.7
    assert "rag" in result['metadata'].get("keywords", [])

def test_direct_query(rag_system):
    """Test handling of direct queries."""
    query = "What is the current time?"
    result = rag_system.process_query(query)
    
    assert result is not None
    assert result.query_type == QueryType.DIRECT
    assert result.should_retrieve is False

def test_calculation_query(rag_system):
    """Test handling of calculation queries."""
    query = "If processing takes 100ms per token, how long for 1000 tokens?"
    result = rag_system.process_query(query)
    
    assert result is not None
    assert result.query_type == QueryType.CALCULATION
    assert result.should_retrieve is False

def test_error_handling(rag_system):
    """Test system error handling."""
    # Test with empty query
    result = rag_system.process_query("")
    assert result is not None
    assert result['metadata'].get("error") is not None
    
    # Test with invalid query type (e.g., integer)
    with pytest.raises(TypeError):
        rag_system.process_query(123) 


def test_complete_pipeline(rag_system, test_queries):
    """Test complete pipeline with multiple queries."""
    evaluator = SystemEvaluator()
    
    # Evaluate system
    results = evaluator.evaluate(
        system=rag_system,
        test_queries=test_queries,
        log_results=True
    )
    
    # Check evaluation metrics
    assert results.accuracy >= 0.7  # At least 70% accurate
    assert results.retrieval_precision >= 0.7
    assert results.error_rate < 0.1  # Less than 10% errors

def test_system_performance(rag_system, test_queries):
    """Test system performance metrics."""
    benchmark = SystemBenchmark()
    
    # Run benchmark
    results = benchmark.run_benchmark(
        system=rag_system,
        test_queries=[q["query"] for q in test_queries],
        num_iterations=2  # Reduced for testing
    )
    
    # Check performance metrics
    assert results["avg_response_time"] < 2.0  # Less than 2 seconds
    assert results["avg_throughput"] > 0.5  # At least 0.5 queries per second

def test_concurrent_queries(rag_system, test_queries):
    """Test system with concurrent queries."""
    from concurrent.futures import ThreadPoolExecutor
    
    def process_query(query):
        return rag_system.process_query(query["query"])
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_query, query) for query in test_queries[:5]]
        results = [future.result() for future in futures]
    
    # Check results
    assert len(results) == 5
    assert all(result is not None for result in results)


def test_vector_store_integration(rag_system, test_documents):
    """Test vector store integration."""
    # Test retrieval query
    query = "What is RAG?"
    result = rag_system.process_query(query)
    
    # Check if relevant documents were retrieved
    assert result['metadata'].get("retrieved_documents") is not None
    assert len(result['metadata']["retrieved_documents"]) > 0
    
    # Check if retrieved content is relevant
    retrieved_text = result['metadata']["retrieved_documents"][0]["content"]
    assert "RAG" in retrieved_text.upper() 
