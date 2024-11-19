import os
from typing import Optional

class Settings:
    """Global settings for the RAG system."""
    
    # Ollama Settings
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "ollama")
    OLLAMA_PORT: int = int(os.getenv("OLLAMA_PORT", "11434"))
    DEFAULT_MODEL: str = os.getenv("OLLAMA_MODEL", "llama2")
    
    # Vector DB Settings
    VECTOR_DB_HOST: str = os.getenv("VECTOR_DB_HOST", "localhost")
    VECTOR_DB_PORT: int = int(os.getenv("VECTOR_DB_PORT", "6333"))
    
    # Application Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    @property
    def ollama_base_url(self) -> str:
        """Get the Ollama API base URL."""
        return f"http://{self.OLLAMA_HOST}:{self.OLLAMA_PORT}"

# Create global settings instance
settings = Settings()