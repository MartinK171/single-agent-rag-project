import time
from config.settings import settings
from config.model_manager import ModelManager

def initialize_system():
    """Initialize the RAG system."""
    print("Initializing RAG Router system...")
    
    # Initialize model manager
    model_manager = ModelManager()
    
    # Initialize default models
    print("Initializing default models...")
    if not model_manager.initialize_default_models():
        print("Failed to initialize default models")
        return False
    
    # Ensure default model is available
    if model_manager.ensure_model_available():
        print(f"Successfully initialized with model: {settings.DEFAULT_MODEL}")
    else:
        print("Failed to initialize model")
        return False
    
    return True

def main():
    """Main entry point for the RAG Router system."""
    if not initialize_system():
        print("System initialization failed")
        return

    print("RAG Router system starting...")
    try:
        while True:
            print("System running and waiting for queries...")
            time.sleep(60)
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    main()