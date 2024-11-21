from typing import List, Dict
import logging
from pathlib import Path
from .loader import DocumentLoader
from .processor import DocumentProcessor
from .chunker import DocumentChunker, Chunk

class DocumentPipeline:
    """Handles the complete document processing pipeline."""

    def __init__(self, 
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 min_chunk_size: int = 100):
        """Initialize the pipeline with all components."""
        self.loader = DocumentLoader()
        self.processor = DocumentProcessor()
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size
        )
        self.logger = logging.getLogger(__name__)

    def process_document(self, file_path: str) -> List[Chunk]:
        """
        Process a single document through the complete pipeline.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of processed chunks
        """
        try:
            # Step 1: Load document
            self.logger.info(f"Loading document: {file_path}")
            document = self.loader.load(file_path)
            
            # Step 2: Process text
            self.logger.info("Processing document content")
            processed_doc = self.processor.process(document)
            
            # Step 3: Create chunks
            self.logger.info("Chunking processed content")
            chunks = self.chunker.chunk_document(processed_doc)
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {str(e)}")
            raise

    def process_directory(self, directory_path: str) -> Dict[str, List[Chunk]]:
        """
        Process all supported documents in a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            Dictionary mapping file paths to their chunks
        """
        directory = Path(directory_path)
        results = {}
        
        # Process each file in directory
        for file_path in directory.glob('*'):
            if file_path.suffix.lower() in self.loader.SUPPORTED_EXTENSIONS:
                try:
                    chunks = self.process_document(str(file_path))
                    results[str(file_path)] = chunks
                except Exception as e:
                    self.logger.error(f"Skipping {file_path}: {str(e)}")
                    continue
                    
        return results