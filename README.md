# Single Agent RAG Router

## Project Overview
This project implements a Single Agent RAG (Retrieval-Augmented Generation) Router system, demonstrating an efficient approach to combining different AI capabilities through a unified routing mechanism. The system intelligently routes queries to appropriate tools based on query type, enhancing response accuracy and capability range.

## Architecture
The system implements a single-agent RAG architecture where a central agent routes queries to different tools based on query analysis:

### Core Components:
- **Query Router**: Central intelligence that determines query type and appropriate tool
- **Vector Store**: Manages knowledge base using Qdrant
- **LLM Integration**: Uses Ollama with Llama2 for generation and reasoning
- **Tool Integration**: Web search, calculator, and vector search capabilities

### Supported Query Types:
1. **Knowledge Retrieval**: Access to stored knowledge base
2. **Web Search**: Current information retrieval
3. **Calculations**: Mathematical operations
4. **Direct Responses**: Immediate LLM-based answers

## Technologies Used
- **FastAPI**: Main application framework
- **Qdrant**: Vector database for knowledge storage
- **Ollama**: Local LLM deployment
- **Docker**: Containerization and deployment
- **Python 3.10+**: Core development language

## Getting Started

### Prerequisites
- Docker Desktop (Windows/macOS)
  - Windows requires WSL 2 (Windows Subsystem for Linux)
  - macOS requires Docker Desktop for Mac
- Git
- Minimum 8GB RAM recommended for running the services
- 4 CPU cores recommended for optimal performance
- Docker Desktop Resource Configuration:
  - Memory: Minimum 10.8 GB
  - CPU: Minimum 10 cores
  - Swap: At least 1 GB
  - Virtual disk limit: At least 64 GB

> **Important Note**: Docker Desktop resource configuration differs slightly between platforms:
> 
> **For Windows**:
> 1. Open Docker Desktop
> 2. Go to Settings (⚙️) 
> 3. Navigate to "Resources" > "WSL 2" on Windows
> 4. Adjust the memory, CPU, and swap limits to at least the minimum values listed above
> 5. Click "Apply & Restart"
>
> **For macOS**:
> 1. Open Docker Desktop
> 2. Go to Settings (⚙️) 
> 3. Navigate to "Resources" > "Advanced"
> 4. Adjust the memory, CPU, and swap limits to at least the minimum values listed above
> 5. Click "Apply & Restart"

### Installation

1. Clone the repository:
```bash
# Windows (PowerShell or Command Prompt)
git clone [repository-url]
cd single-agent-rag-project

# macOS/Linux (Terminal)
git clone [repository-url]
cd single-agent-rag-project
```

2. Start the services:
```bash
# Works the same on all platforms
docker compose up -d
```

3. Verify installation:
```bash
# Windows PowerShell
Invoke-WebRequest http://localhost:8000/health

# Windows Command Prompt
curl http://localhost:8000/health

# macOS/Linux
curl http://localhost:8000/health
```

### Platform-Specific Considerations

#### Windows Users
- Ensure WSL 2 is installed and configured
- Use PowerShell or Command Prompt for running commands
- If using PowerShell and curl commands don't work, use `Invoke-WebRequest` instead
- Line endings in text files should be handled automatically by Git

#### macOS Users
- Ensure Docker Desktop has permission to access your file system
- Terminal commands should work as shown in examples
- If permission issues occur, check Docker Desktop's file sharing settings

#### Common Troubleshooting
- If services fail to start, check Docker Desktop resource allocation
- Ensure all ports (8000, 6333, 11434) are available
- If using antivirus software, you may need to add Docker to the allowed applications

### Installation
1. Clone the repository:
```bash
git clone [repository-url]
cd single-agent-rag-project
```

2. Start the services:
```bash
docker compose up -d
```

3. Verify installation:
```bash
curl http://localhost:8000/health
```

### Configuration
Environment variables can be configured in the `docker-compose.yml` file.

## Initial Setup

### Ollama Model Setup

The project runs Ollama in a Docker container, so you don't need to install the Ollama desktop application. Everything is handled through Docker containers.

After starting the services, you'll need to ensure the required LLM model is available:

1. Pull the Llama2 model:
```bash
# For Windows PowerShell
docker exec -it single-agent-rag-project-ollama-1 ollama pull llama2

# For macOS/Linux Terminal
docker exec -it single-agent-rag-project-ollama-1 ollama pull llama2
```

2. Verify model availability:
```bash
docker exec -it single-agent-rag-project-ollama-1 ollama list
```

> **Note**: 
> - The initial model download might take several minutes depending on your internet connection
> - The Llama2 model is approximately 4GB in size
> - All models are stored in a Docker volume, not on your local system
> - No local Ollama installation is required - everything runs in containers

[Rest of the Ollama setup section remains the same...]

### Populating Vector Stores
After starting the services, you'll need to populate the vector stores with documents before you can perform knowledge retrieval queries.

#### Sample Documents
The project includes sample documents in `data/sample_data/`:
- Technical documentation: 
  - `data/sample_data/technical_docs/rag_systems_overview.txt`
  - `data/sample_data/technical_docs/vector-databases-explained.md`
- Business documentation:
  - `data/sample_data/business_docs/generic-company-profile.md`
  - `data/sample_data/business_docs/rag_market_analysis.txt`

#### Loading Documents Using Postman

1. **Upload Technical Documents**

Using Postman, create a new POST request:
- URL: `http://localhost:8000/upload/files`
- Request Type: POST
- Body: form-data
- Form Fields:
  - Key: `file` (Type: File)
    Value: Select `rag_systems_overview.txt`
  - Key: `store_name` (Type: Text)
    Value: `technical_docs`
  - Key: `metadata` (Type: Text)
    Value: `{"category": "technical", "subject": "rag"}`

Repeat the same process for `vector_databases_explained.txt`, updating the file selection.

2. **Upload Business Documents**

Create another POST request with the same URL but different form data:
- URL: `http://localhost:8000/upload/files`
- Request Type: POST
- Body: form-data
- Form Fields:
  - Key: `file` (Type: File)
    Value: Select `generic-company-profile.md`
  - Key: `store_name` (Type: Text)
    Value: `business_docs`
  - Key: `metadata` (Type: Text)
    Value: `{"category": "business", "type": "company_profile"}`

Repeat for `rag_market_analysis.txt`, updating the file selection.

#### Verify Documents are Loaded

In Postman:
1. Create a GET request to `http://localhost:8000/documents/technical_docs`
2. Create a GET request to `http://localhost:8000/documents/business_docs`

These requests will show you all documents stored in each collection.

#### Test Knowledge Retrieval

In Postman:
1. Create a POST request to `http://localhost:8000/query`
2. Set Content-Type header to `application/json`
3. In the request body (raw, JSON), enter:
```json
{
    "query": "What is RAG and how does it work?"
}
```

For business queries, use:
```json
{
    "query": "What are the current market trends for RAG systems?"
}
```

## API Usage

### Using Postman for API Interactions

#### 1. Query Endpoint
- **URL**: `http://localhost:8000/query`
- **Method**: POST
- **Headers**: 
  - Content-Type: application/json
- **Body** (raw, JSON):
```json
{
    "query": "Your question here"
}
```

Example queries:
```json
{
    "query": "What is RAG and how does it work?"
}
```
```json
{
    "query": "What is 15 multiplied by 25?"
}
```
```json
{
    "query": "What are the latest developments in AI?"
}
```

#### 2. Document Upload
- **URL**: `http://localhost:8000/upload/files`
- **Method**: POST
- **Body**: form-data
- **Form Fields**:
  - file: (Select File)
  - store_name: (technical_docs/business_docs)
  - metadata: (JSON object with document metadata)

#### 3. Document Search
- **URL**: `http://localhost:8000/search/{store_name}`
- **Method**: GET
- **Query Params**: query (your search term)

#### 4. Health Check
- **URL**: `http://localhost:8000/health`
- **Method**: GET

#### 5. List Stores
- **URL**: `http://localhost:8000/stores`
- **Method**: GET

## Testing
Comprehensive testing documentation is available in [TESTING.md](./TESTING.md), including:
- Functional testing results
- System component validation
- Performance observations
- API endpoint testing
- Example queries and responses

## Project Structure
```
SINGLE-AGENT-RAG-PROJECT/
├── config/           # Configuration management
├── src/             # Core source code
│   ├── data_pipeline/   # Document processing
│   ├── integration/     # Service integration
│   ├── query_processing/# Query handling
│   ├── router/         # Query routing logic
│   ├── tools/          # External tools
│   ├── vector_db/      # Vector store management
│   └── main.py         # Application entry point
├── docker-compose.yml  # Service orchestration
└── Dockerfile         # Container configuration
```

## Development Approach
This project was developed with a focus on:
1. Clean, maintainable code structure
2. Robust error handling
3. Clear separation of concerns
4. Extensible architecture
5. Comprehensive documentation

## Future Enhancements
Potential areas for expansion:
1. Advanced query planning
2. Advanced routing capabilities
3. Additional tool integration
4. Performance optimization
5. Enhanced caching mechanisms
