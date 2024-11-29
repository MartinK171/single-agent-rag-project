# API Endpoints Documentation

## Query Endpoints

### 1. Submit Query
```bash
POST /query

# Example 1: Web Search Query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the latest developments in AI?"
}'

# Example Response:
{
    "query": "What are the latest developments in AI?",
    "response": "Based on the recent search results provided, here is an overview of the latest developments in AI...",
    "selected_store": null,
    "query_type": "web_search",
    "confidence": 0.9,
    "success": true,
    "metadata": {
        "processing_path": "question",
        "route_metadata": {
            "reasoning": "Query requires current information"
        }
    }
}

# Example 2: Knowledge Base Query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG and how does it work?"
}'

# Example Response:
{
    "query": "What is RAG and how does it work?",
    "response": "Based on the provided context, RAG stands for Retrieval-Augmented Generation, and it represents a significant advancement in artificial intelligence technology...",
    "selected_store": "technical_docs",
    "query_type": "retrieval",
    "confidence": 0.8,
    "success": true
}

# Example 3: Direct Query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Can you tell me a funny joke?"
}'

# Example 4: Calculation Query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is 15*25?"
}'
```

### 2. Health Check
```bash
GET /health

curl http://localhost:8000/health

# Example Response:
{
    "status": "healthy",
    "vector_stores": ["technical_docs", "business_docs"],
    "llm_status": "available",
    "timestamp": "2024-11-29T12:01:28.287138"
}
```

### 3. Vector Store Management

#### List Available Stores
```bash
GET /stores

curl http://localhost:8000/stores

# Example Response:
{
    "stores": ["technical_docs", "business_docs"],
    "count": 2
}
```

#### Search Documents in Store
```bash
GET /search/{store_name}

curl "http://localhost:8000/search/technical_docs?query=RAG%20implementation"

# Example Response:
{
    "query": "RAG implementation",
    "store_name": "technical_docs",
    "results": [
        {
            "score": 0.85,
            "content": "...",
            "metadata": {
                "source": "documentation",
                "category": "technical"
            }
        }
    ]
}
```

### 4. Document Management

#### Upload Files
```bash
POST /upload/files

# Single file upload
curl -X POST "http://localhost:8000/upload/files" \
  -F "file=@/path/to/document.pdf" \
  -F "store_name=technical_docs" \
  -F "metadata={\"category\":\"technical\",\"source\":\"documentation\"}"

# Multiple files upload
curl -X POST "http://localhost:8000/upload/files" \
  -F "files=@/path/to/doc1.pdf" \
  -F "files=@/path/to/doc2.pdf" \
  -F "store_name=technical_docs"

# Example Response:
{
    "success": true,
    "message": "Successfully processed 2 files",
    "chunks_added": 15,
    "results": [
        {
            "filename": "doc1.pdf",
            "chunk_index": 0,
            "doc_id": "1234-5678",
            "metadata": {
                "source": "documentation",
                "category": "technical"
            }
        }
    ]
}
```

#### Upload Raw Text
```bash
POST /upload/text

curl -X POST "http://localhost:8000/upload/text" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "content=This is the content to store" \
  -d "store_name=technical_docs" \
  -d "metadata={\"category\":\"notes\"}"

# Example Response:
{
    "success": true,
    "message": "Successfully processed text content",
    "chunks_added": 1,
    "results": [
        {
            "chunk_index": 0,
            "doc_id": "abcd-efgh",
            "metadata": {
                "category": "notes",
                "chunk_index": 0,
                "total_chunks": 1
            }
        }
    ]
}
```

### 5. System Management

#### Clear Vector Stores
```bash
POST /clear-stores

curl -X POST "http://localhost:8000/clear-stores"

# Example Response:
{
    "message": "All stores cleared successfully"
}
```

### 6. Document Retrieval

#### List Documents in Store
```bash
GET /documents/{store_name}

curl http://localhost:8000/documents/technical_docs

# Example Response:
{
    "store_name": "technical_docs",
    "document_count": 5,
    "documents": [
        {
            "id": "1234-5678",
            "metadata": {
                "source": "documentation",
                "category": "technical"
            },
            "text": "First 200 characters of the document..."
        }
    ]
}
```
