from typing import Dict, List, Optional
from .store import VectorStore
import logging

class VectorStoreManager:
    """Manages multiple Qdrant collections and coordinates between them."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.stores: Dict[str, VectorStore] = {}
        
    def add_store(self, name: str, collection_name: str):
        """
        Add a new Qdrant collection as a store.
        
        Args:
            name: Name to reference this store
            collection_name: Name of the Qdrant collection
        """
        store = VectorStore(collection_name=collection_name)
        self.stores[name] = store
        self.logger.info(f"Added vector store '{name}' with collection '{collection_name}'")
        
    def get_store(self, name: str) -> Optional[VectorStore]:
        """Get a specific vector store by name."""
        return self.stores.get(name)
        
    def list_stores(self) -> List[str]:
        """List all available store names."""
        return list(self.stores.keys())
        
    def search_all(self, query: str, limit: int = 5) -> Dict[str, List[Dict]]:
        """
        Search across all vector stores.
        
        Args:
            query: The search query
            limit: Maximum number of results per store
            
        Returns:
            Dictionary mapping store names to their search results
        """
        results = {}
        for name, store in self.stores.items():
            try:
                store_results = store.search(query, limit)
                results[name] = store_results
            except Exception as e:
                self.logger.error(f"Error searching store {name}: {str(e)}")
                results[name] = []
        return results
    
    def determine_best_store(self, query: str, store_preference: Optional[str] = None) -> str:
        """
        Determine which store is most appropriate for a given query.
        
        Args:
            query: The search query
            store_preference: Optional preferred store to use
            
        Returns:
            Name of the most appropriate store
        """
        # If store preference is provided and valid, use it
        if store_preference and store_preference in self.stores:
            return store_preference
            
        # Get results from all stores
        all_results = self.search_all(query, limit=1)
        
        # Find store with highest scoring result
        best_store = None
        best_score = -1
        
        for store_name, results in all_results.items():
            if results and results[0]['score'] > best_score:
                best_score = results[0]['score']
                best_store = store_name
                
        # Return best store or first store as fallback
        return best_store or next(iter(self.stores))
    
    def add_document(self, store_name: str, content: str, metadata: Dict) -> str:
        """
        Add a document to a specific store.
        
        Args:
            store_name: Name of the store to add to
            content: Document content
            metadata: Document metadata
            
        Returns:
            Document ID
        """
        store = self.get_store(store_name)
        if not store:
            raise ValueError(f"Store '{store_name}' not found")
            
        doc_ids = store.add_texts([content], [metadata])
        return doc_ids[0]

    def get_store_info(self, store_name: str) -> Dict:
        """
        Get information about a specific store.
        
        Args:
            store_name: Name of the store
            
        Returns:
            Dictionary containing store information
        """
        store = self.get_store(store_name)
        if not store:
            raise ValueError(f"Store '{store_name}' not found")
            
        return {
            "name": store_name,
            "collection_name": store.collection_name,
            "embedding_dimension": 384,  # Using default from the VectorStore class
            "distance": "cosine"
        }