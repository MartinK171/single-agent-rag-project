from typing import Dict, Optional, List
import logging
from dataclasses import dataclass
from .analyzer import QueryAnalyzer, QueryAnalysis
from .templates import ResponseTemplate
from .monitor import QueryMonitor
from .result import ProcessingResult

class QueryProcessor:
    """Processes queries through different paths based on analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.analyzer = QueryAnalyzer()
        self.monitor = QueryMonitor()
        
    def process(self, query: str) -> ProcessingResult:
        """
        Process a query through appropriate processing path.
        
        Args:
            query: Raw query string
            
        Returns:
            ProcessingResult containing processed query and metadata
        """
        try:
            # Start monitoring
            self.monitor.start_processing(query)
            
            # Analyze query
            analysis = self.analyzer.analyze(query)
            
            # Determine processing path
            path = self._determine_path(analysis)
            
            # Process through chosen path
            processed = self._process_through_path(query, path, analysis)
            
            # Get appropriate template
            template = self._get_template(analysis, path)
            
            # Create result
            result = ProcessingResult(
                processed_query=processed,
                analysis=analysis,
                suggested_template=template,
                processing_path=path,
                metadata=self._create_metadata(query, analysis, path)
            )
            
            # Record success
            self.monitor.record_success(query, result)
            
            return result
            
        except Exception as e:
            # Record failure
            self.monitor.record_failure(query, str(e))
            self.logger.error(f"Error processing query: {str(e)}")
            raise
            
    def _determine_path(self, analysis: QueryAnalysis) -> str:
        """Determine appropriate processing path."""
        if analysis.complexity > 0.7:
            return "advanced"
        elif analysis.metadata.get("has_question_mark", False):
            return "question"
        elif analysis.entities:
            return "entity_focused"
        else:
            return "standard"
            
    def _process_through_path(
        self, 
        query: str,
        path: str,
        analysis: QueryAnalysis
    ) -> str:
        """Process query through specified path."""
        processors = {
            "advanced": self._process_advanced,
            "entity_focused": self._process_entity_focused,
            "question": self._process_question,
            "standard": self._process_standard
        }
        
        processor = processors.get(path, self._process_standard)
        return processor(query, analysis)
    
    def _get_template(self, analysis: QueryAnalysis, path: str) -> str:
        """Get appropriate response template."""
        return ResponseTemplate.get_template(path, analysis)
        
    def _create_metadata(
        self,
        query: str,
        analysis: QueryAnalysis,
        path: str
    ) -> Dict:
        """Create processing metadata."""
        return {
            "original_query": query,
            "processing_path": path,
            "analysis_results": analysis.metadata,
            "timestamp": self.monitor.get_current_timestamp()
        }
        
    def _process_advanced(self, query: str, analysis: QueryAnalysis) -> str:
        """Process complex queries."""
        # Advanced query processing logic
        return query
        
    def _process_entity_focused(self, query: str, analysis: QueryAnalysis) -> str:
        """Process entity-focused queries."""
        # Entity-focused processing logic
        return query
        
    def _process_question(self, query: str, analysis: QueryAnalysis) -> str:
        """Process question queries."""
        # Question processing logic
        return query
        
    def _process_standard(self, query: str, analysis: QueryAnalysis) -> str:
        """Process standard queries."""
        # Standard processing logic
        return query
