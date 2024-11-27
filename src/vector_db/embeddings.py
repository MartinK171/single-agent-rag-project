from typing import List
import logging
from sentence_transformers import SentenceTransformer
import numpy as np

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
            self.logger.info("=== Starting embedding generation ===")
            
            if not texts:
                raise ValueError("Empty text list provided")
            if any(not text.strip() for text in texts):
                raise ValueError("Empty string in texts")
                
            self.logger.info(f"Generating embeddings for {len(texts)} texts")
            self.logger.debug(f"First text: {texts[0][:100]}...")
            
            embeddings = self.model.encode(texts)
            
            # Convert to numpy array and then to list
            embeddings_array = np.array(embeddings)
            self.logger.info(f"Generated embeddings with shape: {embeddings_array.shape}")
            
            # Type checking
            self.logger.debug(f"Embeddings type: {type(embeddings)}")
            self.logger.debug(f"First embedding type: {type(embeddings[0])}")
            self.logger.debug(f"First embedding shape: {embeddings[0].shape}")
            
            return embeddings_array
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            self.logger.error(f"Error type: {type(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise