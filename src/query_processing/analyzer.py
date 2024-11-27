from typing import Dict, Optional
import logging
from dataclasses import dataclass, field
import re

@dataclass
class QueryAnalysis:
    """Results of query analysis."""
    complexity: float = 0.0  # 0-1 score of query complexity
    keywords: list[str] = field(default_factory=list)   # Important keywords from query
    entities: list[str] = field(default_factory=list)  # Named entities found
    topic: Optional[str] = None  # Main topic if identified
    metadata: Dict = field(default_factory=dict)   # Additional analysis info

class QueryAnalyzer:
    """Analyzes queries for better processing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze a query to determine its characteristics.
        
        Args:
            query: The query string to analyze
            
        Returns:
            QueryAnalysis containing analysis results
        """
        try:
            # Extract keywords
            keywords = self._extract_keywords(query)
            
            # Find entities
            entities = self._find_entities(query)
            
            # Calculate complexity
            complexity = self._calculate_complexity(query)
            
            # Identify topic
            topic = self._identify_topic(query)
            
            return QueryAnalysis(
                complexity=complexity,
                keywords=keywords,
                entities=entities,
                topic=topic,
                metadata={
                    "length": len(query),
                    "word_count": len(query.split()),
                    "has_question_mark": "?" in query
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing query: {str(e)}")
            return QueryAnalysis(
                complexity=0.5,
                keywords=[],
                entities=[],
                topic=None,
                metadata={"error": str(e)}
            )
            
    def _extract_keywords(self, query: str) -> list[str]:
        """Extract important keywords from query."""
        # Use regex to extract words, ignoring punctuation
        words = re.findall(r'\b\w+\b', query.lower())
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'is'}
        return [w for w in words if w not in stopwords]
    
    def _find_entities(self, query: str) -> list[str]:
        """Find named entities in query."""
        # Simple entity extraction: extract all words with all uppercase letters (e.g., 'RAG')
        return re.findall(r'\b[A-Z]{2,}\b', query)
    
    def _calculate_complexity(self, query: str) -> float:
        """Calculate query complexity score."""
        # Basic complexity scoring
        factors = [
            len(query) > 100,  # Long query
            "?" in query,      # Contains question
            "and" in query,    # Multiple parts
            "or" in query      # Alternative paths
        ]
        return sum(factors) / len(factors)
    
    def _identify_topic(self, query: str) -> Optional[str]:
        """Identify main topic of query."""
        # Basic topic identification
        return None
