from typing import Dict
import logging
from datetime import datetime
from .result import ProcessingResult 

class QueryMonitor:
    """Monitors and logs query processing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "processing_times": [],
            "path_usage": {},
            "errors": []
        }
        
    def start_processing(self, query: str):
        """Record start of query processing."""
        self.current_start = datetime.now()
        self.metrics["total_queries"] += 1
        
    def record_success(self, query: str, result: ProcessingResult):
        """Record successful query processing."""
        self.metrics["successful_queries"] += 1
        processing_time = (datetime.now() - self.current_start).total_seconds()
        self.metrics["processing_times"].append(processing_time)
        
        # Record path usage
        path = result.processing_path  # Access attribute directly
        self.metrics["path_usage"][path] = self.metrics["path_usage"].get(path, 0) + 1
        
        # Log success
        self.logger.info(
            f"Query processed successfully: {query[:100]}... "
            f"(time: {processing_time:.2f}s)"
        )
        
    def record_failure(self, query: str, error: str):
        """Record failed query processing."""
        self.metrics["failed_queries"] += 1
        self.metrics["errors"].append({
            "query": query,
            "error": error,
            "timestamp": self.get_current_timestamp()
        })
        
        # Log failure
        self.logger.error(
            f"Query processing failed: {query[:100]}... "
            f"Error: {error}"
        )
        
    def get_metrics(self) -> Dict:
        """Get current metrics."""
        return {
            **self.metrics,
            "average_processing_time": self._calculate_average_time(),
            "success_rate": self._calculate_success_rate()
        }
        
    def _calculate_average_time(self) -> float:
        """Calculate average processing time."""
        times = self.metrics["processing_times"]
        return sum(times) / len(times) if times else 0
        
    def _calculate_success_rate(self) -> float:
        """Calculate success rate."""
        total = self.metrics["total_queries"]
        if total == 0:
            return 0
        return self.metrics["successful_queries"] / total
        
    def get_current_timestamp(self) -> str:
        """Get current timestamp string."""
        return datetime.now().isoformat()
