from typing import List
import logging
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    """Handles text to vector embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
    def generate(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            if not texts:
                raise ValueError("Empty text list provided")
            if any(not text.strip() for text in texts):
                raise ValueError("Empty string in texts")
                
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise