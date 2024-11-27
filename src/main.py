from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from integration.rag_pipeline import RAGPipeline
from langchain_ollama import OllamaLLM
from vector_db.manager import VectorStoreManager
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Single Agent RAG API")

# Initialize the vector store manager
vector_store_manager = VectorStoreManager()

try:
    # Add two different Qdrant collections
    vector_store_manager.add_store(
        name="technical_docs",
        collection_name="technical_collection"
    )
    vector_store_manager.add_store(
        name="business_docs",
        collection_name="business_collection"
    )
    logger.info("Successfully initialized vector stores")
except Exception as e:
    logger.error(f"Error initializing vector stores: {str(e)}")
    raise

try:
    # Initialize the LLM
    llm = OllamaLLM(
        model=os.getenv("OLLAMA_MODEL", "llama2"),
        host=os.getenv("OLLAMA_HOST", "ollama"),
        port=int(os.getenv("OLLAMA_PORT", "11434"))
    )
    logger.info("Successfully initialized LLM")
except Exception as e:
    logger.error(f"Error initializing LLM: {str(e)}")
    raise

try:
    # Initialize the RAGPipeline
    rag_pipeline = RAGPipeline(llm=llm, vector_store_manager=vector_store_manager)
    logger.info("Successfully initialized RAG pipeline")
except Exception as e:
    logger.error(f"Error initializing RAG pipeline: {str(e)}")
    raise

class QueryRequest(BaseModel):
    query: str
    store_preference: str | None = None  # Optional preferred vector store

class QueryResponse(BaseModel):
    query: str
    response: str
    selected_store: str | None = None
    query_type: str
    confidence: float
    success: bool
    metadata: dict | None = None

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "vector_stores": vector_store_manager.list_stores(),
        "llm_status": "available"
    }

@app.get("/stores")
async def list_stores():
    """List available vector stores."""
    return {
        "stores": vector_store_manager.list_stores()
    }

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """Handle incoming queries."""
    logger.info(f"Received query: {request.query}")
    
    try:
        # Process the query using RAGPipeline
        result = rag_pipeline.process_query(
            query=request.query,
            store_preference=request.store_preference
        )
        
        return QueryResponse(
            query=request.query,
            response=result.get('response', ''),
            selected_store=result.get('selected_store'),
            query_type=result.get('query_type', ''),
            confidence=result.get('confidence', 0.0),
            success=result.get('success', False),
            metadata=result.get('metadata', {})
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

class DocumentInput(BaseModel):
    content: str
    metadata: dict
    store_name: str

@app.post("/documents")
async def add_document(document: DocumentInput):
    """Add a document to a specific vector store."""
    try:
        store = vector_store_manager.get_store(document.store_name)
        if not store:
            raise HTTPException(
                status_code=404,
                detail=f"Vector store '{document.store_name}' not found"
            )
            
        # Add document to the specified store
        doc_ids = store.add_texts(
            texts=[document.content],
            metadata=[document.metadata]
        )
        
        return {
            "success": True,
            "document_id": doc_ids[0],
            "store": document.store_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error adding document: {str(e)}"
        )

@app.post("/clear-stores")
async def clear_stores():
    """Clear all vector store collections."""
    try:
        for store_name in vector_store_manager.list_stores():
            store = vector_store_manager.get_store(store_name)
            if store:
                store.client.delete_collection(store.collection_name)
                store._init_collection()
        return {"message": "All stores cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)