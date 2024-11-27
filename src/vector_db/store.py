from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
import logging
import numpy as np  # Add this import
from .embeddings import EmbeddingGenerator

class VectorStore:
    """Handles vector storage and retrieval."""
    
    def __init__(self, 
                 collection_name: str = "test_collection",
                 embedding_generator: Optional[EmbeddingGenerator] = None):
        """Initialize vector store."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)  # Set to DEBUG for more detailed logs
        
        self.collection_name = collection_name
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333"))
        )
        
        # Initialize collection
        self._init_collection()

    def _init_collection(self):
        """Initialize Qdrant collection if it doesn't exist."""
        try:
            if not self.client.collection_exists(self.collection_name):
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
                
            # Log collection info
            collection_info = self.client.get_collection(self.collection_name)
            self.logger.debug(f"Collection info: {collection_info}")
            
        except Exception as e:
            self.logger.error(f"Error initializing collection: {str(e)}")
            raise

    def add_texts(self, texts: List[str], metadata: List[Dict]) -> List[str]:
        """Add texts to vector store."""
        try:
            self.logger.info("=== Starting document addition ===")
            self.logger.info(f"Adding {len(texts)} texts to collection {self.collection_name}")
            self.logger.debug(f"First text content: {texts[0][:100]}...")  # Log first 100 chars
            self.logger.debug(f"Metadata: {metadata[0]}")

            # Generate embeddings
            self.logger.info("Generating embeddings...")
            embeddings = self.embedding_generator.generate(texts)
            # Convert to numpy array if not already
            embeddings = np.array(embeddings)
            self.logger.info(f"Generated embeddings with shape: {embeddings.shape}")

            # Create points
            self.logger.info("Creating points...")
            import uuid
            points = []
            for i, (text, embedding, meta) in enumerate(zip(texts, embeddings, metadata)):
                point_id = str(uuid.uuid4())
                # Add the text content to the metadata
                meta['content'] = text
                try:
                    point = models.PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload=meta
                    )
                    points.append(point)
                    self.logger.debug(f"Created point {i+1}/{len(texts)} with ID: {point_id}")
                    self.logger.debug(f"Point vector shape: {len(embedding)}")
                except Exception as e:
                    self.logger.error(f"Error creating point {i+1}: {str(e)}")
                    raise

            # Upload to Qdrant
            self.logger.info(f"Uploading {len(points)} points to Qdrant...")
            upsert_result = self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=points
            )
            self.logger.info("Upload completed")

            # Verify points were added
            count = self.client.count(collection_name=self.collection_name)
            self.logger.info(f"Collection now contains {count.count} points")

            # Verify search works
            test_query = "test search"
            self.logger.info(f"Testing search with query: {test_query}")
            test_results = self.search(test_query)
            self.logger.info(f"Test search found {len(test_results)} results")

            self.logger.info("=== Document addition completed successfully ===")
            return [str(p.id) for p in points]

        except Exception as e:
            self.logger.error(f"Error adding texts: {str(e)}")
            self.logger.error(f"Error type: {type(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        try:
            query_embedding = self.embedding_generator.generate([query])[0]
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit,
                score_threshold=0.3  # Lower threshold
            )
            
            self.logger.debug(f"Raw search results: {results}")
            
            formatted_results = []
            for hit in results:
                formatted_results.append({
                    'score': float(hit.score),
                    'text': hit.payload.get('content', ''),  # Ensure content is retrieved
                    'metadata': {k:v for k,v in hit.payload.items() if k != 'content'}
                })
                
            return formatted_results

        except Exception as e:
            self.logger.error(f"Error searching: {str(e)}")
            raise
            
    def get_collection_info(self) -> Dict:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            count = self.client.count(self.collection_name)
            return {
                "name": self.collection_name,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance,
                "total_points": count.count
            }
        except Exception as e:
            self.logger.error(f"Error getting collection info: {str(e)}")
            raise