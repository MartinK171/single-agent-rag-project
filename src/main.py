from fastapi import FastAPI
from pydantic import BaseModel
from integration.rag_pipeline import RAGPipeline
from langchain_ollama import OllamaLLM  # Added import
from src.vector_db.store import VectorStore  # Added import

app = FastAPI()

# Initialize the vector store
vector_store = VectorStore(
    collection_name='test_init'
)

# Initialize the LLM
llm = OllamaLLM(
    model="llama2",
    host="ollama",
    port=11434
)

# Initialize the RAGPipeline with llm and vector_store
rag_pipeline = RAGPipeline(llm=llm, vector_store=vector_store)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    response: str
    success: bool
    query_type: str  
    confidence: float

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """Handle incoming queries."""
    print(f"Received query: {request.query}")
    try:
        # Process the query using RAGPipeline
        result = rag_pipeline.process_query(request.query)
        return QueryResponse(
            query=request.query,
            response=result.get('response', ''),
            success=result.get('success', False),
            query_type=result.get('query_type', ''),
            confidence=result.get('confidence', 0.0)
        )
    except Exception as e:
        print(f"Error processing query: {e}")
        return QueryResponse(
            query=request.query,
            response="An error occurred while processing your query.",
            success=False,
            query_type="",
            confidence=0.0
        )
