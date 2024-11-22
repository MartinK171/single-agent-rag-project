from typing import List, Dict
import logging
from pathlib import Path
from .loader import DocumentLoader
from .processor import DocumentProcessor
from .chunker import DocumentChunker, Chunk

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class DocumentPipeline:
    """Handles the complete document processing pipeline."""

    def __init__(self, 
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 min_chunk_size: int = 100):
        """Initialize the pipeline with all components."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing DocumentPipeline")
        
        self.logger.debug("Creating DocumentLoader")
        self.loader = DocumentLoader()
        
        self.logger.debug("Creating DocumentProcessor")
        self.processor = DocumentProcessor()
        
        self.logger.debug("Creating DocumentChunker")
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size
        )

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
            self.logger.debug(f"Loading document: {file_path}")
            document = self.loader.load(file_path)
            self.logger.debug("Document loaded successfully")
            
            # Step 2: Process text
            self.logger.debug("Processing document content")
            processed_doc = self.processor.process(document)
            self.logger.debug("Document processed successfully")
            
            # Step 3: Create chunks
            self.logger.debug("Chunking processed content")
            chunks = self.chunker.chunk_document(processed_doc)
            self.logger.debug(f"Document chunked successfully into {len(chunks)} chunks")
            
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
        
        for file_path in directory.glob('*'):
            if file_path.suffix.lower() in self.loader.SUPPORTED_EXTENSIONS:
                try:
                    self.logger.debug(f"Processing file: {file_path}")
                    chunks = self.process_document(str(file_path))
                    results[str(file_path)] = chunks
                except Exception as e:
                    self.logger.error(f"Skipping {file_path}: {str(e)}")
                    continue
                    
        return results