from typing import Dict, List
import re
import html
import logging

class DocumentProcessor:
    """Handles processing and normalization of document content."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def process(self, document: Dict) -> Dict:
        """
        Process document content with various cleaning and normalization steps.
        
        Args:
            document: Dict containing 'content' and 'metadata'
            
        Returns:
            Dict with processed content and updated metadata
        """
        try:
            content = document['content']
            doc_type = document['metadata']['type']

            # Apply processing steps sequentially
            processed_content = content
            processed_content = self._remove_html_tags(processed_content)
            processed_content = self._normalize_whitespace(processed_content)
            processed_content = self._normalize_quotes(processed_content)
            processed_content = self._normalize_unicode(processed_content)

            # Update document with processed content
            processed_doc = document.copy()
            processed_doc['content'] = processed_content
            processed_doc['metadata']['processed'] = True
            processed_doc['metadata']['processing_steps'] = [
                'remove_html',
                'normalize_whitespace',
                'normalize_quotes',
                'normalize_unicode'
            ]

            return processed_doc

        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise

    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags and decode HTML entities."""
        # First decode HTML entities
        text = html.unescape(text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove whitespace at the beginning and end
        text = text.strip()
        return text

    def _normalize_quotes(self, text: str) -> str:
        """Normalize different types of quotes."""
        # Convert smart quotes to regular quotes
        replacements = {
            '"': '"',  # Smart double quotes
            '"': '"',
            ''': "'",  # Smart single quotes
            ''': "'",
            '«': '"',  # Guillemets
            '»': '"',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters."""
        # Convert common Unicode characters to ASCII equivalents
        replacements = {
            '–': '-',  # en dash
            '—': '-',  # em dash
            '…': '...',  # ellipsis
            '\u200b': '',  # zero-width space
            '\xa0': ' ',  # non-breaking space
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def process_batch(self, documents: List[Dict]) -> List[Dict]:
        """
        Process multiple documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of processed documents
        """
        processed_docs = []
        for doc in documents:
            try:
                processed_doc = self.process(doc)
                processed_docs.append(processed_doc)
            except Exception as e:
                self.logger.error(f"Error processing document in batch: {str(e)}")
                # Continue processing other documents even if one fails
                continue
        return processed_docs