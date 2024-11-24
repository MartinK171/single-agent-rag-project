from typing import Dict
from dataclasses import dataclass
from .analyzer import QueryAnalysis

@dataclass
class ProcessingResult:
    """Result of query processing."""
    processed_query: str
    analysis: QueryAnalysis
    suggested_template: str
    processing_path: str
    metadata: Dict
