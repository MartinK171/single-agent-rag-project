import pytest
import time
from typing import List, Dict

from src.integration.rag_pipeline import RAGPipeline
from tests.utils.benchmarks import SystemBenchmark

@pytest.fixture
def performance_queries() -> List[str]:
    """Generate test queries for performance testing."""
    return [
        "What is RAG?",
        "How does vector search work?",
        "Explain embedding generation",
        # Add more queries as needed
    ] * 10  # Multiply for load testing

def test_load_performance(rag_system, performance_queries):
    """Test system performance under load."""
    benchmark = SystemBenchmark()
    
    results = benchmark.run_benchmark(
        system=rag_system,
        test_queries=performance_queries,
        num_iterations=5,
        cooldown=1.0
    )
    
    # Check performance metrics
    assert results["avg_response_time"] < 2.0
    assert results["avg_memory_usage"] < 1024  # MB
    assert results["avg_cpu_usage"] < 80  # percent

def test_concurrent_performance(rag_system, performance_queries):
    """Test performance with concurrent requests."""
    from concurrent.futures import ThreadPoolExecutor
    
    def process_query(query):
        return rag_system.process_query(query)
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_query, q) for q in performance_queries]
        results = [future.result() for future in futures]
    
    total_time = time.time() - start_time
    
    # Check results
    assert len(results) == len(performance_queries)
    assert (total_time / len(performance_queries)) < 0.5  # Average time per query


def test_memory_leaks(rag_system, performance_queries):
    """Test for memory leaks during extended use."""
    import gc
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run multiple query batches
    for _ in range(5):
        for query in performance_queries:
            _ = rag_system.process_query(query)
        gc.collect()  # Force garbage collection
    
    final_memory = process.memory_info().rss / 1024 / 1024
    memory_increase = final_memory - initial_memory
    
    # Check memory usage
    assert memory_increase < 100  # MB
