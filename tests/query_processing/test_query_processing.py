import pytest
from src.query_processing.analyzer import QueryAnalyzer, QueryAnalysis
from src.query_processing.processor import QueryProcessor
from src.query_processing.result import ProcessingResult
from src.query_processing.templates import ResponseTemplate
from src.query_processing.monitor import QueryMonitor

@pytest.fixture
def analyzer():
    return QueryAnalyzer()

@pytest.fixture
def processor():
    return QueryProcessor()

@pytest.fixture
def monitor():
    return QueryMonitor()

# QueryAnalyzer Tests
def test_analyzer_basic(analyzer):
    """Test basic query analysis."""
    query = "What is RAG in machine learning?"
    analysis = analyzer.analyze(query)

    assert isinstance(analysis, QueryAnalysis)
    assert analysis.keywords == ["what", "rag", "machine", "learning"]
    assert 0 <= analysis.complexity <= 1
    assert "has_question_mark" in analysis.metadata
    assert analysis.entities == ["RAG"] 

def test_analyzer_empty_query(analyzer):
    """Test analyzer with empty query."""
    analysis = analyzer.analyze("")

    assert isinstance(analysis, QueryAnalysis)
    assert analysis.keywords == []
    assert analysis.complexity == 0.0

def test_analyzer_complex_query(analyzer):
    """Test analyzer with complex query."""
    query = "Can you explain how RAG works and compare it with traditional approaches?"
    analysis = analyzer.analyze(query)

    assert analysis.complexity > 0.5 
    assert "rag" in analysis.keywords
    assert analysis.metadata["has_question_mark"]

# QueryProcessor Tests
def test_processor_basic(processor):
    """Test basic query processing."""
    query = "What is RAG?"
    result = processor.process(query)

    assert isinstance(result, ProcessingResult)
    assert result.processed_query
    assert result.analysis
    assert result.processing_path in ["standard", "question"]

def test_processor_paths(processor):
    """Test different processing paths."""
    # Test question path
    question_result = processor.process("What is RAG?")
    assert question_result.processing_path == "question"

    standard_result = processor.process("Tell me about RAG")
    assert standard_result.processing_path == "entity_focused"

def test_processor_error_handling(processor):
    """Test processor error handling."""
    with pytest.raises(Exception):
        processor.process(None) 

# ResponseTemplate Tests
def test_template_selection():
    """Test template selection."""
    analysis = QueryAnalysis(
        complexity=0.8,
        keywords=["rag"],
        entities=[],
        topic=None,
        metadata={}
    )
    
    template = ResponseTemplate.get_template("advanced", analysis)
    assert "Detailed Response" in template
    assert "Complexity Analysis" in template

def test_template_customization():
    """Test template customization based on analysis."""
    analysis = QueryAnalysis(
        complexity=0.3,
        keywords=["rag"],
        entities=["RAG"],
        topic=None,
        metadata={}
    )
    
    template = ResponseTemplate.get_template("entity_focused", analysis)
    assert "Entity Details" in template

# QueryMonitor Tests
def test_monitor_metrics(monitor):
    """Test monitoring metrics."""
    # Record some activity
    monitor.start_processing("test query")
    monitor.record_success("test query", ProcessingResult(
        processed_query="processed",
        analysis=QueryAnalysis(
            complexity=0.5,
            keywords=["test"],
            entities=[],
            topic=None,
            metadata={}
        ),
        suggested_template="standard",
        processing_path="standard",
        metadata={}
    ))
    
    metrics = monitor.get_metrics()
    assert metrics["total_queries"] == 1
    assert metrics["successful_queries"] == 1
    assert metrics["failed_queries"] == 0
    assert len(metrics["processing_times"]) == 1

def test_monitor_failure_tracking(monitor):
    """Test failure monitoring."""
    monitor.start_processing("bad query")
    monitor.record_failure("bad query", "Test error")
    
    metrics = monitor.get_metrics()
    assert metrics["failed_queries"] == 1
    assert len(metrics["errors"]) == 1
    assert metrics["errors"][0]["error"] == "Test error"

def test_monitor_success_rate(monitor):
    """Test success rate calculation."""
    # Record mix of successes and failures
    for i in range(3):
        monitor.start_processing(f"query_{i}")
        if i % 2 == 0:
            monitor.record_success(f"query_{i}", ProcessingResult(
                processed_query="processed",
                analysis=QueryAnalysis(
                    complexity=0.5,
                    keywords=["test"],
                    entities=[],
                    topic=None,
                    metadata={}
                ),
                suggested_template="standard",
                processing_path="standard",
                metadata={}
            ))
        else:
            monitor.record_failure(f"query_{i}", "Test error")
    
    metrics = monitor.get_metrics()
    assert metrics["total_queries"] == 3
    assert metrics["successful_queries"] == 2
    assert metrics["failed_queries"] == 1
    assert metrics["success_rate"] == pytest.approx(2/3)

# Integration Tests
def test_complete_processing_flow(processor):
    """Test complete processing flow from query to result."""
    query = "What is RAG and how does it work with LLMs?"
    
    result = processor.process(query)
    
    # Check all components worked together
    assert isinstance(result, ProcessingResult)
    assert result.analysis.keywords
    assert result.suggested_template
    assert result.processing_path
    assert result.metadata["original_query"] == query

def test_error_propagation(processor):
    """Test error handling throughout the pipeline."""
    with pytest.raises(Exception):
        processor.process(None)
