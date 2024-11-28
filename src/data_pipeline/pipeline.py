from .loader import DocumentLoader
from .chunker import DocumentChunker

class DocumentPipeline:
    def __init__(self):
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker()

    def process_document(self, file_path: str):
        """Process document with metadata."""
        try:
            # Load document with metadata
            doc_with_metadata = self.loader.load(file_path)
            
            # Verify content exists
            if not doc_with_metadata.get('content'):
                raise ValueError("No content extracted from document")
            
            # Prepare for chunking
            document = {
                'content': doc_with_metadata['content'],
                'metadata': doc_with_metadata.get('metadata', {})
            }
            
            # Create chunks
            chunks = self.chunker.chunk_document(document)
            
            if not chunks:
                raise ValueError("Document chunking produced no results")
                
            return chunks, doc_with_metadata['metadata']
            
        except Exception as e:
            raise Exception(f"Error processing document {file_path}: {str(e)}")