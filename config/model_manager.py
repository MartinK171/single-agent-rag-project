import requests
from typing import List, Optional
from .settings import settings

class ModelManager:
    """Manages Ollama models for the RAG system."""
    
    def __init__(self, base_url: Optional[str] = None):
        # Use provided URL or get from settings
        self.base_url = base_url or settings.ollama_base_url
        
    def list_models(self) -> List[str]:
        """List all available models."""
        try:
            # Make GET request to Ollama API to list models
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                # Extract model names from response
                return [model["name"] for model in response.json()["models"]]
            return []
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a specific model."""
        try:
            # Make POST request to Ollama API to pull model
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name}
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Error pulling model {model_name}: {e}")
            return False
    
    def ensure_model_available(self, model_name: str = None) -> bool:
        """Ensure a specific model is available, pulling if needed."""
        model_name = model_name or settings.DEFAULT_MODEL
        available_models = self.list_models()
        
        if model_name not in available_models:
            print(f"Model {model_name} not found. Pulling...")
            return self.pull_model(model_name)
        return True

    def initialize_default_models(self) -> bool:
        """Pull default models (Llama 2 and Mistral)."""
        models_to_pull = ["llama2", "mistral"]
        success = True
        
        for model in models_to_pull:
            print(f"Ensuring {model} is available...")
            if not self.ensure_model_available(model):
                print(f"Failed to pull {model}")
                success = False
        
        return success

    def switch_model(self, new_model: str) -> bool:
        """Switch to a different model."""
        if self.ensure_model_available(new_model):
            print(f"Switching to model: {new_model}")
            settings.DEFAULT_MODEL = new_model
            return True
        return False
    
    def process_query(self, query: str) -> str:
        """Process the query and return a response."""
        # For now, return a placeholder response
        return f"Processed query: {query}"