from typing import List, Dict, Optional
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from .embeddings import EmbeddingGenerator
import os

class VectorStore:
    def __init__(self, 
                 collection_name: str = "test_collection",  # Changed default name
                 embedding_generator: Optional[EmbeddingGenerator] = None):
        self.logger = logging.getLogger(__name__)
        self.collection_name = collection_name
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        
        # Use environment variables for configuration
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        
        # Initialize Qdrant client
        self.client = QdrantClient(qdrant_host, port=qdrant_port)
        
        # Initialize collection
        self._init_collection()
        
    def _init_collection(self):
        """Initialize Qdrant collection if it doesn't exist."""
        try:
            # Check if collection exists
            if not self.client.collection_exists(self.collection_name):
                # Create new collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=384,  # MiniLM embedding size
                        distance=models.Distance.COSINE
                    )
                )
                self.logger.info(f"Created collection: {self.collection_name}")
            else:
                self.logger.info(f"Using existing collection: {self.collection_name}")
                
        except Exception as e:
            self.logger.error(f"Error initializing collection: {str(e)}")
            raise

    def add_texts(self, texts: List[str], metadata: List[Dict]) -> List[str]:
        """Add texts to vector store."""
        try:
            # Generate embeddings
            embeddings = self.embedding_generator.generate(texts)
            
            # Create points with proper IDs
            import uuid
            points = []
            for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
                point_id = str(uuid.uuid4())
                point = models.PointStruct(
                    id=point_id,
                    vector=embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                    payload=meta
                )
                points.append(point)
            
            # Upload to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=points
            )
            
            return [str(p.id) for p in points]
            
        except Exception as e:
            self.logger.error(f"Error adding texts: {str(e)}")
            raise
            
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search for similar texts.
        
        Args:
            query: Text to search for
            limit: Maximum number of results
            
        Returns:
            List of results with scores and metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate([query])[0]
            
            # Search in Qdrant
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            
            # Format results
            return [{
                'score': hit.score,
                'metadata': hit.payload,
            } for hit in results]
            
        except Exception as e:
            self.logger.error(f"Error searching: {str(e)}")
            raise