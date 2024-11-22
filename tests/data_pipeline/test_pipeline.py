import pytest
import tempfile
import os
import logging
from src.data_pipeline import DocumentPipeline

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@pytest.fixture
def test_content():
    """Fixture to provide test document content."""
    return """
    This is a test document. It has multiple sentences.
    This is another paragraph with some content.
    """

@pytest.fixture
def test_files():
    """Fixture to provide test files configuration."""
    return {
        "doc1.txt": "This is document one.",
        "doc2.txt": "This is document two.",
        "ignored.xyz": "This should be ignored."
    }

def test_pipeline_basic(test_content, capfd):
    """Test basic pipeline functionality."""
    print("\n=== Starting Pipeline Test ===")
    print(f"\nTest content:\n{test_content}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write(test_content)
        
        print(f"\nCreated test file at: {test_file}")
        
        pipeline = DocumentPipeline(
            chunk_size=50,
            chunk_overlap=10,
            min_chunk_size=20
        )
        
        print("\nProcessing document...")
        chunks = pipeline.process_document(test_file)
        
        # Verify results
        assert chunks is not None
        assert len(chunks) > 0
        
        # Test chunk properties
        for chunk in chunks:
            assert len(chunk.text) > 0
            assert chunk.start_char >= 0
            assert chunk.end_char > chunk.start_char
            assert chunk.metadata['total_chunks'] == len(chunks)
        
        # Print results
        print(f"\nPipeline produced {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i + 1}/{len(chunks)}:")
            print(f"Text: '{chunk.text}'")
            print(f"Start: {chunk.start_char}, End: {chunk.end_char}")
            print(f"Metadata: {chunk.metadata}")
            print("-" * 50)
        
        # Capture and verify output
        out, _ = capfd.readouterr()
        assert "Starting Pipeline Test" in out
        assert "Processing document" in out

def test_pipeline_directory(test_files):
    """Test directory processing."""
    print("\n=== Starting Directory Processing Test ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        for filename, content in test_files.items():
            with open(os.path.join(temp_dir, filename), "w") as f:
                f.write(content)
        
        pipeline = DocumentPipeline()
        results = pipeline.process_directory(temp_dir)
        
        # Verify results
        assert len(results) == 2  # Should only process .txt files
        assert "ignored.xyz" not in [os.path.basename(p) for p in results.keys()]
        
        # Test each processed file
        for file_path, chunks in results.items():
            assert file_path.endswith('.txt')
            assert len(chunks) > 0
            
            # Test chunk properties
            for chunk in chunks:
                assert len(chunk.text) > 0
                # Verify basic metadata exists
                assert chunk.metadata is not None
                assert 'processed' in chunk.metadata
                assert 'processing_steps' in chunk.metadata
                assert 'chunk_index' in chunk.metadata
                
                # Verify metadata values
                assert isinstance(chunk.metadata['chunk_index'], int)
                assert chunk.metadata['processed'] is True
                assert isinstance(chunk.metadata['processing_steps'], list)