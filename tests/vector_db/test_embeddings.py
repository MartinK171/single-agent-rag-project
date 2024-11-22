import pytest
import numpy as np
from src.vector_db import EmbeddingGenerator

@pytest.fixture
def embedding_generator():
    return EmbeddingGenerator()

@pytest.fixture
def sample_texts():
    return [
        "This is the first test document",
        "Here is another document",
        "And a third one for testing"
    ]

def test_embedding_generator_init():
    """Test embedding generator initialization."""
    generator = EmbeddingGenerator()
    assert generator is not None
    assert generator.model is not None

def test_generate_embeddings(embedding_generator, sample_texts):
    """Test generating embeddings from texts."""
    embeddings = embedding_generator.generate(sample_texts)
    
    # Check basic properties
    assert len(embeddings) == len(sample_texts)
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(len(emb) == 384 for emb in embeddings)  # MiniLM model dimension
    
    # Check embeddings are different for different texts
    emb_arrays = [np.array(emb) for emb in embeddings]
    for i in range(len(emb_arrays)):
        for j in range(i + 1, len(emb_arrays)):
            similarity = np.dot(emb_arrays[i], emb_arrays[j])
            assert similarity != 1.0  # Different texts should have different embeddings

def test_generate_single_text(embedding_generator):
    """Test generating embedding for a single text."""
    text = "This is a test document"
    embeddings = embedding_generator.generate([text])
    
    assert len(embeddings) == 1
    assert isinstance(embeddings[0], list)
    assert len(embeddings[0]) == 384

def test_error_handling(embedding_generator):
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError): 
        embedding_generator.generate([])
        
    with pytest.raises(ValueError):
        embedding_generator.generate([""])