from src.router.types import QueryType, RouterResponse

def test_query_type_enum():
    """Test QueryType enumeration values."""
    assert QueryType.RETRIEVAL.value == "retrieval"
    assert QueryType.DIRECT.value == "direct"
    assert QueryType.CALCULATION.value == "calculation"
    assert QueryType.CLARIFICATION.value == "clarification"

def test_router_response_creation():
    """Test RouterResponse creation and attributes."""
    response = RouterResponse(
        query_type=QueryType.RETRIEVAL,
        confidence=0.9,
        should_retrieve=True,
        retrieval_query="modified query",
        metadata={"reasoning": "test reason"}
    )
    
    assert response.query_type == QueryType.RETRIEVAL
    assert response.confidence == 0.9
    assert response.should_retrieve is True
    assert response.retrieval_query == "modified query"
    assert response.metadata == {"reasoning": "test reason"}