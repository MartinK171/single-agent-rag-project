import pytest
import logging
from pathlib import Path

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return {
        "vector_store": {
            "collection_name": "test_collection",
            "dimension": 384  # Dimension of embeddings
        },
        "evaluation": {
            "min_accuracy": 0.7,
            "max_response_time": 2.0
        }
    }

@pytest.fixture(scope="session")
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent.parent