from typing import List, Dict
import logging
from dataclasses import dataclass

@dataclass
class Chunk:
    """Represents a chunk of text with metadata."""
    text: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict

class DocumentChunker:
    """Handles splitting documents into chunks with configurable strategies."""

    def __init__(self, 
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 min_chunk_size: int = 100):
        """
        Initialize the chunker with configuration.

        Args:
            chunk_size: Target size for each chunk
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum size for a chunk
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.logger = logging.getLogger(__name__)

    def chunk_document(self, document: Dict) -> List[Chunk]:
        """
        Split a document into chunks.
        Args:
            document: Dictionary containing 'content' and 'metadata'

        Returns:
            List of Chunk objects
        """
        try:
            self.logger.debug("Starting to chunk document")
            text = document['content']
            chunks = []
            
            # Handle empty or very short text
            if len(text) <= self.min_chunk_size:
                return [Chunk(
                    text=text,
                    index=0,
                    start_char=0,
                    end_char=len(text),
                    metadata={**document['metadata'], 'chunk_index': 0}
                )]

            # Split text into chunks
            start = 0
            chunk_index = 0

            while start < len(text):
                end = start + self.chunk_size
                
                # Don't go beyond text length
                if end > len(text):
                    end = len(text)
                
                # Create chunk
                chunk_text = text[start:end]
                chunk = Chunk(
                    text=chunk_text,
                    index=chunk_index,
                    start_char=start,
                    end_char=end,
                    metadata={
                        **document['metadata'],
                        'chunk_index': chunk_index,
                        'total_chunks': -1
                    }
                )
                chunks.append(chunk)
                
                # Move start position for next chunk
                start = end - self.chunk_overlap
                chunk_index += 1

                # Break if we've reached the end
                if end >= len(text):
                    break

            # Update total chunks in metadata
            for chunk in chunks:
                chunk.metadata['total_chunks'] = len(chunks)

            return chunks

        except Exception as e:
            self.logger.error(f"Error chunking document: {str(e)}")
            raise

    def chunk_batch(self, documents: List[Dict]) -> List[List[Chunk]]:
        """
        Chunk multiple documents.

        Args:
            documents: List of document dictionaries

        Returns:
            List of lists of chunks (one list per document)
        """
        chunked_docs = []
        for doc in documents:
            try:
                chunks = self.chunk_document(doc)
                chunked_docs.append(chunks)
            except Exception as e:
                self.logger.error(f"Error chunking document in batch: {str(e)}")
                continue
        return chunked_docs