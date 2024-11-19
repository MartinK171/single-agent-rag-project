# Single-Agent RAG Router System

## Project Overview
A flexible, on-premise Retrieval-Augmented Generation (RAG) system with intelligent routing capabilities.

## Features
- Local LLM using Ollama
- Flexible document processing
- Intelligent query routing
- On-premise deployment

## Prerequisites
- Docker
- Docker Compose
- Python 3.10+

## Installation

### Local Setup
```bash
# Clone the repository
git clone https://github.com/MartinK171/single-agent-rag-project.git
cd single-agent-rag-project

# Install dependencies
pip install -r requirements.txt

# Pull Ollama models
ollama pull llama2