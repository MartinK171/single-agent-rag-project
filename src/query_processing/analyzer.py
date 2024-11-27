from typing import Dict, Optional
import logging
from dataclasses import dataclass, field
import re

@dataclass
class QueryAnalysis:
    complexity: float = 0.0
    keywords: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    topic: Optional[str] = None
    temporal_aspects: Dict = field(default_factory=dict)
    calculation_aspects: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


class QueryAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, query: str) -> QueryAnalysis:
        try:
            keywords = self._extract_keywords(query)
            entities = self._find_entities(query)
            complexity = self._calculate_complexity(query)
            temporal_aspects = self._analyze_temporal_aspects(query)
            calculation_aspects = self._analyze_calculation_aspects(query)
            
            return QueryAnalysis(
                complexity=complexity,
                keywords=keywords,
                entities=entities,
                temporal_aspects=temporal_aspects,
                calculation_aspects=calculation_aspects,
                metadata={
                    "length": len(query),
                    "word_count": len(query.split()),
                    "has_question_mark": "?" in query
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing query: {str(e)}")
            return QueryAnalysis()

    def _analyze_temporal_aspects(self, query: str) -> Dict:
        time_indicators = {
            'recent': True, 'latest': True, 'last': True,
            'current': True, 'now': True, 'today': True
        }
        return {
            "requires_current_info": any(i in query.lower() for i in time_indicators),
            "temporal_indicators": [i for i in time_indicators if i in query.lower()]
        }

    def _analyze_calculation_aspects(self, query: str) -> Dict:
        calc_indicators = ['+', '-', '*', '/', '%', 'calculate', 'sum', 'multiply']
        return {
            "requires_calculation": any(i in query for i in calc_indicators),
            "has_numbers": bool(re.search(r'\d+', query))
        }
    
    def _identify_topic(self, query: str) -> Optional[str]:
        """Identify main topic of query."""
        topic_indicators = {
            'rag': 'retrieval_augmentation',
            'math': 'calculation',
            'calculate': 'calculation',
            'news': 'web_search',
            'latest': 'web_search',
            'recent': 'web_search',
            'document': 'retrieval',
            'docs': 'retrieval'
        }
        
        words = query.lower().split()
        for word in words:
            if word in topic_indicators:
                return topic_indicators[word]
        return None
            
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
