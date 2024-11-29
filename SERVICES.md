# Auxiliary Services Documentation

## Qdrant Vector Database

### Accessing Qdrant Dashboard
The Qdrant Dashboard is available at: `http://localhost:6333/dashboard`

Through this interface you can:
- View collections
- Monitor collection metrics
- Inspect points and payloads
- Test search queries
- Monitor system health

### Qdrant REST API
Qdrant's REST API is available at: `http://localhost:6333`

Common operations:
```bash
# List collections
curl http://localhost:6333/collections

# Get collection info
curl http://localhost:6333/collections/technical_docs

# Check collection existence 
curl http://localhost:6333/collections/technical_docs/exists
```

## Ollama Service

### Accessing Ollama
Ollama service is available at: `http://localhost:11434`

Common operations:
```bash
# List available models
curl http://localhost:11434/api/tags

# Check model status
curl http://localhost:11434/api/show \
  -d '{"name": "llama2"}'
```

### Managing Ollama Models
If you need to manage models directly from host machine:
```bash
# Pull a model
docker exec -it single-agent-rag-project-ollama-1 ollama pull llama2

# List models
docker exec -it single-agent-rag-project-ollama-1 ollama list

# Remove a model
docker exec -it single-agent-rag-project-ollama-1 ollama rm llama2
```

## Container Management

### Viewing Logs
```bash
# View logs for all services
docker compose logs

# Follow logs for specific service
docker compose logs -f single-agent-rag

# View Qdrant logs
docker compose logs qdrant

# View Ollama logs
docker compose logs ollama
```

### Container Status
```bash
# Check running containers
docker compose ps

# Check container health
docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
```

### Resource Usage
```bash
# Check container resource usage
docker stats

# Check specific container
docker stats single-agent-rag-project-qdrant-1
```

### Container Management
```bash
# Restart specific service
docker compose restart qdrant

# Stop all services
docker compose down

# Start all services
docker compose up -d

# Rebuild and start
docker compose up -d --build
```

## System Monitoring

### Memory Usage
Monitor service memory usage:
```bash
# Check memory usage for all containers
docker stats --format "table {{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}"
```

### Performance Monitoring
```bash
# View container performance stats
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
```

### Network Inspection
```bash
# List networks
docker network ls

# Inspect network
docker network inspect single-agent-rag-project_app-network
```