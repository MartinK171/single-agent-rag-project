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
                # Find end position for current chunk
                end = self._find_chunk_end(text, start)
                
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
                        'total_chunks': -1  # Will be updated later
                    }
                )
                chunks.append(chunk)
                
                # Move start position for next chunk
                start = end - self.chunk_overlap
                chunk_index += 1

            # Update total chunks in metadata
            for chunk in chunks:
                chunk.metadata['total_chunks'] = len(chunks)

            return chunks

        except Exception as e:
            self.logger.error(f"Error chunking document: {str(e)}")
            raise

    def _find_chunk_end(self, text: str, start: int) -> int:
        """
        Find the end position for a chunk, trying to break at sentence boundary.
        """
        # Calculate the ideal end position
        ideal_end = start + self.chunk_size
        
        # If we're near the end of text, return the end
        if ideal_end >= len(text):
            return len(text)
            
        # Look for sentence boundaries near ideal end
        window_start = max(ideal_end - 100, start + self.min_chunk_size)
        window_end = min(ideal_end + 100, len(text))
        text_window = text[window_start:window_end]
        
        # Common sentence endings
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        
        # Find the best sentence boundary
        best_end = None
        for ending in sentence_endings:
            # Look for sentence ending in the window
            pos = text_window.find(ending)
            if pos != -1:
                absolute_pos = window_start + pos + len(ending)
                if best_end is None or abs(absolute_pos - ideal_end) < abs(best_end - ideal_end):
                    best_end = absolute_pos
        
        # If no good sentence boundary found, use ideal end
        if best_end is None:
            # Try to break at word boundary instead
            text_window = text[ideal_end-10:ideal_end+10]
            space_positions = [i for i, char in enumerate(text_window) if char.isspace()]
            if space_positions:
                # Find the closest space to the ideal end
                closest_space = min(space_positions, key=lambda x: abs(x - 10))
                best_end = ideal_end - 10 + closest_space
            else:
                best_end = ideal_end
                
        return best_end

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