from typing import Union, Dict
from pathlib import Path
import os
import json
import logging
import docx
import PyPDF2
from datetime import datetime

class DocumentLoader:
    """Document loader with metadata extraction."""
    
    SUPPORTED_EXTENSIONS = {
        '.txt': 'text',
        '.json': 'json',
        '.md': 'markdown',
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.doc': 'doc',
        '.rtf': 'rtf',
        '.html': 'html',
        '.htm': 'html',
        '.xml': 'xml'
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load(self, file_path: Union[str, Path]) -> Dict:
        """Load document with metadata extraction."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            extension = file_path.suffix.lower()
            if extension not in self.SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported file type: {extension}")
            
            # Load content based on file type
            content = self._load_content(file_path)
            
            # Generate metadata
            metadata = self._generate_metadata(file_path, content)
            
            return {
                'content': content,
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error loading document {file_path}: {str(e)}")
            raise

    def _load_content(self, file_path: Path) -> str:
        """Load content based on file type."""
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.pdf':
                return self._load_pdf(file_path)
            elif extension in ['.docx', '.doc']:
                return self._load_docx(file_path)
            elif extension == '.txt':
                return self._load_text(file_path)
            elif extension == '.json':
                return self._load_json(file_path)
            elif extension == '.md':
                return self._load_markdown(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
                    
        except Exception as e:
            self.logger.error(f"Error loading content: {str(e)}")
            raise

    def _load_pdf(self, file_path: Path) -> str:
        """Load content from PDF file."""
        try:
            self.logger.debug(f"Starting to read PDF file: {file_path}")
            
            with open(file_path, 'rb') as file:
                # Create PDF reader object
                reader = PyPDF2.PdfReader(file)
                
                if len(reader.pages) == 0:
                    self.logger.warning(f"PDF file {file_path} has no pages")
                    return ""
                
                # Extract text from all pages
                text_content = []
                for page in reader.pages:
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            text_content.append(text.strip())
                    except Exception as e:
                        self.logger.error(f"Error extracting text from page: {str(e)}")
                        continue
                
                # Join all text
                final_content = "\n\n".join(text_content)
                
                if not final_content.strip():
                    self.logger.warning(f"No text content extracted from PDF: {file_path}")
                    return ""
                    
                self.logger.debug(f"Successfully extracted {len(final_content)} characters from PDF")
                return final_content
                
        except Exception as e:
            self.logger.error(f"Error reading PDF {file_path}: {str(e)}")
            raise

    def _generate_metadata(self, file_path: Path, content: str) -> Dict:
        """Generate metadata for document."""
        stats = os.stat(file_path)
        
        return {
            'file_info': {
                'filename': file_path.name,
                'extension': file_path.suffix.lower(),
                'size': stats.st_size,
                'created_at': datetime.fromtimestamp(stats.st_ctime).isoformat(),
                'modified_at': datetime.fromtimestamp(stats.st_mtime).isoformat()
            },
            'content_info': {
                'char_count': len(content),
                'word_count': len(content.split()),
                'line_count': len(content.splitlines())
            },
            'processing': {
                'processor_version': '1.0',
                'processing_time': datetime.now().isoformat()
            }
        }

    def _load_docx(self, file_path: Path) -> str:
        """Load content from DOCX file."""
        doc = docx.Document(file_path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])

    def _load_text(self, file_path: Path) -> str:
        """Load content from text file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def _load_json(self, file_path: Path) -> str:
        """Load content from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data.get('content', json.dumps(data))

    def _load_markdown(self, file_path: Path) -> str:
        """Load content from Markdown file."""
        return self._load_text(file_path)