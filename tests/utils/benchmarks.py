from typing import Dict, List, Optional
import time
import psutil
import logging
from datetime import datetime
import platform

class SystemBenchmark:
    """Performs system benchmarking."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def run_benchmark(self,
                     system,
                     test_queries: List[str],
                     num_iterations: int = 3,
                     cooldown: float = 1.0) -> Dict:
        """
        Run benchmark tests on the system.
        
        Args:
            system: The RAG system to benchmark
            test_queries: List of test queries
            num_iterations: Number of times to run tests
            cooldown: Time to wait between iterations
            
        Returns:
            Dictionary containing benchmark results
        """
        try:
            self.logger.info("Starting benchmark...")
            
            results = {
                'response_times': [],
                'memory_usage': [],
                'cpu_usage': [],
                'throughput': []
            }
            
            # Run iterations
            for i in range(num_iterations):
                self.logger.info(f"Running iteration {i+1}/{num_iterations}")
                
                iteration_results = self._run_iteration(system, test_queries)
                
                # Record results
                results['response_times'].extend(iteration_results['response_times'])
                results['memory_usage'].append(iteration_results['memory_usage'])
                results['cpu_usage'].append(iteration_results['cpu_usage'])
                results['throughput'].append(iteration_results['throughput'])
                
                # Cooldown between iterations
                if i < num_iterations - 1:
                    time.sleep(cooldown)
            
            # Calculate final metrics
            final_results = self._calculate_metrics(results)
            
            self.logger.info("Benchmark complete")
            self._log_benchmark_results(final_results)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error during benchmark: {str(e)}")
            raise
    
    def _run_iteration(self, system, queries: List[str]) -> Dict:
        """Run a single benchmark iteration."""
        start_time = time.time()
        response_times = []
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        # Process queries
        for query in queries:
            query_start = time.time()
            system.process_query(query)
            response_times.append(time.time() - query_start)
        
        # Calculate metrics
        total_time = time.time() - start_time
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        return {
            'response_times': response_times,
            'memory_usage': end_memory - start_memory,
            'cpu_usage': psutil.cpu_percent(interval=1.0),
            'throughput': len(queries) / total_time
        }
    
    def _calculate_metrics(self, results: Dict) -> Dict:
        """Calculate final benchmark metrics."""
        return {
            'avg_response_time': sum(results['response_times']) / len(results['response_times']),
            'max_response_time': max(results['response_times']),
            'min_response_time': min(results['response_times']),
            'avg_memory_usage': sum(results['memory_usage']) / len(results['memory_usage']),
            'avg_cpu_usage': sum(results['cpu_usage']) / len(results['cpu_usage']),
            'avg_throughput': sum(results['throughput']) / len(results['throughput']),
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'system_info': self._get_system_info()
            }
        }
    
    def _get_system_info(self) -> Dict:
        """Get system information."""
        return {
            'cpu_count': psutil.cpu_count(),
            'total_memory': psutil.virtual_memory().total / (1024 * 1024),  # MB
            'platform': platform.platform()
        }
    
    def _log_benchmark_results(self, results: Dict):
        """Log benchmark results."""
        self.logger.info("=== Benchmark Results ===")
        self.logger.info(f"Average Response Time: {results['avg_response_time']:.3f}s")
        self.logger.info(f"Average Memory Usage: {results['avg_memory_usage']:.1f}MB")
        self.logger.info(f"Average CPU Usage: {results['avg_cpu_usage']:.1f}%")
        self.logger.info(f"Average Throughput: {results['avg_throughput']:.2f} queries/sec")