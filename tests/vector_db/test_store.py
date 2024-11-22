import pytest
import time
from src.vector_db import VectorStore

@pytest.fixture
def vector_store():
    """Create a fresh vector store for each test."""
    store = VectorStore(collection_name="test_collection")
    yield store
    # Cleanup after test
    try:
        store.client.delete_collection(collection_name="test_collection")
    except:
        pass

@pytest.fixture
def sample_documents():
    return [
        "The quick brown fox jumps over the lazy dog",
        "A man walks his dog in the park",
        "The cat sleeps on the windowsill"
    ]

@pytest.fixture
def sample_metadata():
    return [
        {"source": "doc1.txt", "category": "test"},
        {"source": "doc2.txt", "category": "test"},
        {"source": "doc3.txt", "category": "test"}
    ]

def test_store_initialization():
    """Test vector store initialization."""
    store = VectorStore(collection_name="test_init")
    assert store is not None
    assert store.collection_name == "test_init"
    collections = store.client.get_collections().collections
    assert any(c.name == "test_init" for c in collections)

def test_add_and_search(vector_store, sample_documents, sample_metadata):
    """Test adding documents and searching."""
    # Add documents
    ids = vector_store.add_texts(sample_documents, sample_metadata)
    assert len(ids) == len(sample_documents)
    
    # Wait for Qdrant to process
    time.sleep(2)
    
    # Search
    query = "dog walking"
    results = vector_store.search(query, limit=2)
    
    # Check results
    assert len(results) > 0
    assert all('score' in result for result in results)
    assert all('metadata' in result for result in results)

def test_search_with_empty_collection(vector_store):
    """Test searching in empty collection."""
    results = vector_store.search("test query")
    assert len(results) == 0

def test_search_with_limit(vector_store, sample_documents, sample_metadata):
    """Test search with different limits."""
    # Add documents
    ids = vector_store.add_texts(sample_documents, sample_metadata)
    
    # Wait for Qdrant to process
    time.sleep(2)
    
    # Test different limits
    limit_1 = vector_store.search("dog", limit=1)
    limit_2 = vector_store.search("dog", limit=2)
    
    assert len(limit_1) <= 1
    assert len(limit_2) <= 2