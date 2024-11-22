import pytest
import tempfile
import os
import logging
from src.data_pipeline import DocumentLoader, DocumentProcessor, DocumentChunker

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def test_file():
    """Fixture to provide a temporary test file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("Test content")
        yield test_file

@pytest.fixture
def test_document():
    """Fixture to provide a test document."""
    return {
        'content': 'Test <b>content</b>',
        'metadata': {'type': 'text'}
    }

@pytest.fixture
def test_chunker_document():
    """Fixture to provide a document for chunking."""
    return {
        'content': 'This is a test. This is another test.',
        'metadata': {'type': 'text'}
    }

def test_loader(test_file):
    """Test DocumentLoader separately."""
    logger.info("Testing DocumentLoader...")
    
    loader = DocumentLoader()
    result = loader.load(test_file)
    
    assert result is not None
    assert 'content' in result
    assert 'metadata' in result
    assert result['content'] == "Test content"
    logger.info("DocumentLoader test passed")

def test_processor(test_document):
    """Test DocumentProcessor separately."""
    logger.info("Testing DocumentProcessor...")
    
    processor = DocumentProcessor()
    result = processor.process(test_document)
    
    assert result is not None
    assert 'content' in result
    assert 'metadata' in result
    assert '<b>' not in result['content']
    logger.info("DocumentProcessor test passed")

def test_chunker(test_chunker_document):
    """Test DocumentChunker separately."""
    logger.info("Testing DocumentChunker...")
    
    chunker = DocumentChunker(
        chunk_size=10,
        chunk_overlap=2,
        min_chunk_size=5
    )
    
    logger.info("Starting chunk_document method...")
    chunks = chunker.chunk_document(test_chunker_document)
    
    assert chunks is not None
    assert len(chunks) > 0
    
    # Test chunk properties
    for chunk in chunks:
        assert len(chunk.text) > 0
        assert chunk.start_char >= 0
        assert chunk.end_char > chunk.start_char
        assert 'chunk_index' in chunk.metadata
        assert 'total_chunks' in chunk.metadata
    
    logger.info(f"Got {len(chunks)} chunks")