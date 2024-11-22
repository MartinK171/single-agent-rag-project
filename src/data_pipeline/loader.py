from typing import Union, List, Dict
from pathlib import Path
import os
import json
import logging

class DocumentLoader:
    """Handles loading of documents from various file formats."""
    
    SUPPORTED_EXTENSIONS = {
        '.txt': 'text',
        '.json': 'json',
        '.md': 'markdown'
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load(self, file_path: Union[str, Path]) -> Dict:
        """
        Load a document from a file path.
        
        Args:
            file_path: Path to the document
            
        Returns:
            dict: Document content and metadata
            {
                'content': str,
                'metadata': {
                    'source': str,
                    'type': str,
                    'size': int,
                    'created_at': str
                }
            }
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        extension = file_path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {extension}")
            
        try:
            loader_method = getattr(self, f"_load_{self.SUPPORTED_EXTENSIONS[extension]}")
            content = loader_method(file_path)
            
            return {
                'content': content,
                'metadata': self._get_metadata(file_path)
            }
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {str(e)}")
            raise
    
    def _load_text(self, file_path: Path) -> str:
        """Load content from text file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _load_json(self, file_path: Path) -> str:
        """Load content from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # If JSON contains a 'content' field, use that, otherwise stringify the whole JSON
            return data.get('content', json.dumps(data))
    
    def _load_markdown(self, file_path: Path) -> str:
        """Load content from Markdown file."""
        return self._load_text(file_path)
    
    def _get_metadata(self, file_path: Path) -> Dict:
        """Extract metadata from file."""
        stats = os.stat(file_path)
        return {
            'source': str(file_path),
            'type': self.SUPPORTED_EXTENSIONS[file_path.suffix.lower()],
            'size': stats.st_size,
            'created_at': stats.st_ctime
        }