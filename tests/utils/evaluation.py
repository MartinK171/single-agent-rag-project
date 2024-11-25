from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import logging
from datetime import datetime

@dataclass
class EvaluationResult:
    """Results of system evaluation."""
    accuracy: float
    retrieval_precision: float
    avg_confidence: float
    avg_response_time: float
    error_rate: float
    metadata: Dict

class SystemEvaluator:
    """Evaluates RAG system performance."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def evaluate(self, 
                system,
                test_queries: List[Dict],
                log_results: bool = True) -> EvaluationResult:
        """
        Evaluate system performance against test queries.
        
        Args:
            system: The RAG system to evaluate
            test_queries: List of test query dictionaries
            log_results: Whether to log detailed results
            
        Returns:
            EvaluationResult with computed metrics
        """
        try:
            results = []
            total_time = 0
            errors = 0
            
            # Process each test query
            for query_data in test_queries:
                try:
                    # Time the response
                    start_time = datetime.now()
                    response = system.process_query(query_data['query'])
                    response_time = (datetime.now() - start_time).total_seconds()
                    
                    # Record results
                    results.append({
                        'query_id': query_data['id'],
                        'expected_type': query_data['expected_type'],
                        'actual_type': response.get('query_type', 'unknown'),
                        'confidence': response.get('confidence', 0.0),
                        'response_time': response_time,
                        'success': self._check_success(response, query_data)
                    })
                    
                    total_time += response_time
                    
                except Exception as e:
                    self.logger.error(f"Error processing query {query_data['id']}: {str(e)}")
                    errors += 1
            
            # Calculate metrics
            metrics = self._calculate_metrics(results, len(test_queries), total_time, errors)
            
            # Log results if requested
            if log_results:
                self._log_evaluation_results(metrics, results)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            raise
            
    def _check_success(self, response: Dict, expected: Dict) -> bool:
        """Check if response matches expected behavior."""
        return (
            response.get('query_type', '').lower() == expected['expected_type'].lower() and
            response.get('confidence', 0.0) >= expected['expected_behavior']['minimum_confidence'] and
            response.get('should_retrieve', False) == expected['expected_behavior']['should_retrieve']
        )
    
    def _calculate_metrics(self, 
                         results: List[Dict],
                         total_queries: int,
                         total_time: float,
                         errors: int) -> EvaluationResult:
        """Calculate evaluation metrics."""
        if not results:
            return EvaluationResult(
                accuracy=0.0,
                retrieval_precision=0.0,
                avg_confidence=0.0,
                avg_response_time=0.0,
                error_rate=1.0,
                metadata={"error": "No results"}
            )
        
        # Calculate metrics
        successful = sum(1 for r in results if r['success'])
        accuracy = successful / total_queries
        
        retrieval_queries = [r for r in results 
                           if r['expected_type'] == 'retrieval']
        retrieval_precision = sum(1 for r in retrieval_queries 
                                if r['success']) / len(retrieval_queries) if retrieval_queries else 0
        
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        avg_response_time = total_time / total_queries
        error_rate = errors / total_queries
        
        return EvaluationResult(
            accuracy=accuracy,
            retrieval_precision=retrieval_precision,
            avg_confidence=avg_confidence,
            avg_response_time=avg_response_time,
            error_rate=error_rate,
            metadata={
                "total_queries": total_queries,
                "successful_queries": successful,
                "errors": errors,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def _log_evaluation_results(self, metrics: EvaluationResult, results: List[Dict]):
        """Log detailed evaluation results."""
        self.logger.info("=== Evaluation Results ===")
        self.logger.info(f"Accuracy: {metrics.accuracy:.2%}")
        self.logger.info(f"Retrieval Precision: {metrics.retrieval_precision:.2%}")
        self.logger.info(f"Average Confidence: {metrics.avg_confidence:.2f}")
        self.logger.info(f"Average Response Time: {metrics.avg_response_time:.3f}s")
        self.logger.info(f"Error Rate: {metrics.error_rate:.2%}")