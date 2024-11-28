from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import tempfile
import os
import json
import logging
from pathlib import Path
from datetime import datetime

from src.data_pipeline import DocumentPipeline
from src.vector_db.manager import VectorStoreManager
from langchain_ollama import OllamaLLM
from integration.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Single Agent RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the vector store manager
try:
    logger.info("Initializing vector store manager...")
    vector_store_manager = VectorStoreManager()

    # Add vector stores
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
        "llm_status": "available",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stores")
async def list_stores():
    """List available vector stores."""
    try:
        stores = vector_store_manager.list_stores()
        return {
            "stores": stores,
            "count": len(stores)
        }
    except Exception as e:
        logger.error(f"Error listing stores: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing stores: {str(e)}"
        )

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

@app.get("/search/{store_name}")
async def search_documents(store_name: str, query: str):
    """Search documents in a store."""
    try:
        store = vector_store_manager.get_store(store_name)
        if not store:
            raise HTTPException(
                status_code=404,
                detail=f"Vector store '{store_name}' not found"
            )

        # Search the store
        results = store.search(query, limit=5)

        return {
            'query': query,
            'store_name': store_name,
            'results': results
        }

    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error searching documents: {str(e)}"
        )

@app.post("/upload/files")
async def upload_files(
    files: List[UploadFile] = File(...),
    store_name: str = Form(...),
    metadata: Optional[str] = Form(None)
):
    """
    Upload and process multiple files.
    
    Args:
        files: List of files to upload
        store_name: Target vector store name
        metadata: Optional JSON string with additional metadata
    """
    logger.info(f"Starting file upload process for {len(files)} files to store '{store_name}'")
    
    try:
        # Validate store exists
        store = vector_store_manager.get_store(store_name)
        if not store:
            logger.error(f"Vector store '{store_name}' not found")
            raise HTTPException(
                status_code=404,
                detail=f"Vector store '{store_name}' not found"
            )

        # Parse metadata if provided
        parsed_metadata = {}
        if metadata:
            try:
                parsed_metadata = json.loads(metadata)
                logger.debug(f"Parsed metadata: {parsed_metadata}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid metadata JSON: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid metadata JSON: {str(e)}"
                )

        # Process each file
        results = []
        for file in files:
            logger.info(f"Processing file: {file.filename}")
            
            # Validate file size
            file_size = 0
            try:
                file_size = len(await file.read())
                await file.seek(0)  # Reset file pointer
                logger.debug(f"File size: {file_size} bytes")
                
                if file_size == 0:
                    logger.warning(f"Empty file received: {file.filename}")
                    continue
                    
            except Exception as e:
                logger.error(f"Error checking file size: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Error processing file {file.filename}: {str(e)}"
                )

            # Create temporary file
            suffix = Path(file.filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                try:
                    # Read and write content
                    content = await file.read()
                    logger.debug(f"Read {len(content)} bytes from {file.filename}")
                    
                    temp_file.write(content)
                    temp_file.flush()
                    logger.debug(f"Wrote content to temporary file: {temp_file.name}")

                    # Initialize pipeline and process document
                    pipeline = DocumentPipeline()
                    logger.debug(f"Processing document through pipeline: {temp_file.name}")
                    
                    chunks, doc_metadata = pipeline.process_document(temp_file.name)
                    logger.info(f"Document processed into {len(chunks)} chunks")

                    # Combine with user metadata
                    doc_metadata.update(parsed_metadata)
                    
                    # Store chunks
                    for chunk in chunks:
                        chunk_metadata = {
                            **doc_metadata,
                            'chunk_info': {
                                'index': chunk.index,
                                'start_char': chunk.start_char,
                                'end_char': chunk.end_char,
                                'total_chunks': len(chunks)
                            },
                            'file_info': {
                                'filename': file.filename,
                                'content_type': file.content_type,
                                'size': file_size,
                                'upload_timestamp': datetime.now().isoformat()
                            }
                        }
                        
                        # Add to vector store
                        try:
                            logger.debug(f"Adding chunk {chunk.index} to vector store")
                            ids = store.add_texts(
                                texts=[chunk.text],
                                metadata=[chunk_metadata]
                            )
                            
                            results.append({
                                'filename': file.filename,
                                'chunk_index': chunk.index,
                                'doc_id': ids[0],
                                'metadata': chunk_metadata
                            })
                            
                        except Exception as e:
                            logger.error(f"Error adding chunk to vector store: {str(e)}")
                            raise

                except Exception as e:
                    logger.error(f"Error processing file {file.filename}: {str(e)}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error processing file {file.filename}: {str(e)}"
                    )
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(temp_file.name)
                        logger.debug(f"Cleaned up temporary file: {temp_file.name}")
                    except Exception as e:
                        logger.warning(f"Error removing temp file: {str(e)}")

        # Return results
        return {
            'success': True,
            'message': f'Successfully processed {len(files)} files',
            'chunks_added': len(results),
            'results': results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in upload endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing uploads: {str(e)}"
        )

@app.post("/upload/text")
async def upload_text(
    content: str = Form(...),
    store_name: str = Form(...),
    metadata: Optional[str] = Form(None)
):
    """
    Upload and process raw text content.
    
    Args:
        content: Text content to process
        store_name: Target vector store name
        metadata: Optional JSON string with additional metadata
    """
    logger.info(f"Starting text upload process to store '{store_name}'")
    
    try:
        # Validate store exists
        store = vector_store_manager.get_store(store_name)
        if not store:
            logger.error(f"Vector store '{store_name}' not found")
            raise HTTPException(
                status_code=404,
                detail=f"Vector store '{store_name}' not found"
            )

        # Parse metadata if provided
        parsed_metadata = {}
        if metadata:
            try:
                parsed_metadata = json.loads(metadata)
                logger.debug(f"Parsed metadata: {parsed_metadata}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid metadata JSON: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid metadata JSON: {str(e)}"
                )

        # Create document for pipeline
        document = {
            "content": content,
            "metadata": {
                **parsed_metadata,
                "upload_timestamp": datetime.now().isoformat(),
                "content_type": "text/plain"
            }
        }

        # Process document
        pipeline = DocumentPipeline()
        chunks = pipeline.chunker.chunk_document(document)
        logger.info(f"Text content processed into {len(chunks)} chunks")

        results = []
        # Add chunks to vector store
        for chunk in chunks:
            chunk_metadata = {
                **document["metadata"],
                "chunk_index": chunk.index,
                "total_chunks": len(chunks),
                "start_char": chunk.start_char,
                "end_char": chunk.end_char
            }
            
            ids = store.add_texts(
                texts=[chunk.text],
                metadata=[chunk_metadata]
            )
            
            results.append({
                "chunk_index": chunk.index,
                "doc_id": ids[0],
                "metadata": chunk_metadata
            })

        return {
            "success": True,
            "message": "Successfully processed text content",
            "chunks_added": len(results),
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing text upload: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing text upload: {str(e)}"
        )

@app.get("/documents/{store_name}")
async def get_documents(store_name: str):
    """List all documents in a store with their metadata."""
    try:
        store = vector_store_manager.get_store(store_name)
        if not store:
            raise HTTPException(
                status_code=404,
                detail=f"Vector store '{store_name}' not found"
            )

        # Get all documents from the store
        results = store.client.scroll(
            collection_name=store.collection_name,
            limit=100  # Can be adjusted
        )[0]  # Get first batch of results

        # Format the response
        documents = []
        for result in results:
            documents.append({
                'id': result.id,
                'metadata': result.payload,
                'text': result.payload.get('text', '')[:200] + '...'  # First 200 chars
            })

        return {
            'store_name': store_name,
            'document_count': len(documents),
            'documents': documents
        }

    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving documents: {str(e)}"
        )

@app.post("/clear-stores")
async def clear_stores():
    """Clear all vector store collections."""
    try:
        logger.info("Starting clear stores operation")
        for store_name in vector_store_manager.list_stores():
            store = vector_store_manager.get_store(store_name)
            if store:
                logger.info(f"Clearing store: {store_name}")
                store.client.delete_collection(store.collection_name)
                store._init_collection()
        logger.info("Successfully cleared all stores")
        return {"message": "All stores cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing stores: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)