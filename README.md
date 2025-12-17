# dApp Code Generation

This repository demonstrates a code generation app powered by Large Language Models (LLMs) and Knowledge Graphs, specifically designed for Rust and Substrate framework development. The system uses Retrieval Augmented Generation (RAG) with specialized knowledge graphs to provide accurate and contextual code suggestions.

## System Overview

### Core Concepts

The system combines several powerful concepts to provide accurate code generation:

1. **Knowledge Graphs (KG)**:
   - Structured representation of programming knowledge
   - Captures relationships between code concepts
   - Enables semantic understanding of code context
   - Built from curated documentation and repositories

2. **Retrieval Augmented Generation (RAG)**:
   - Uses knowledge graphs for context-aware code generation
   - Improves accuracy by providing relevant context to LLMs
   - Reduces hallucination in generated code
   - Enables domain-specific knowledge integration

3. **Multi-Framework Support**:
   - Specialized knowledge graphs for different frameworks
   - Independent processing of each framework's context
   - Extensible to new programming frameworks

### Architecture Design

The system is designed to be cloud-agnostic and easily deployable anywhere:

1. **Storage Layer**:
   - Currently uses S3 for knowledge graph storage
   - Can be adapted to any object storage (GCS, Azure Blob, MinIO)
   - Knowledge graphs are portable and framework-independent
   - Simple file-based format for easy migration

2. **Compute Layer**:
   - Stateless API design
   - No cloud-specific compute requirements
   - Can run on any infrastructure (cloud, on-premise, local)
   - Horizontally scalable

3. **Caching Layer**:
   - Redis for performance optimization
   - Replaceable with any caching solution
   - Optional component for smaller deployments

4. **Monitoring Layer**:
   - Langfuse for observability
   - Modular design allows different monitoring solutions
   - Optional for basic deployments

## Features

- AI-powered code generation for Rust and Substrate
- Knowledge Graph-enhanced context understanding
- RAG (Retrieval Augmented Generation) system
- Multiple specialized knowledge bases (Substrate, Ink!, Solidity, Rust)
- Observability and performance tracking with Langfuse
- Redis caching for improved response times

## Project Structure
```
dApp-codegen/
├── api/
│   ├── app.py
│   └── utils.py
├── caching/
│   ├── __init__.py
│   └── redis_cache.py
├── demos/
│   ├── gradio/
│   │   ├── app_v2.py
│   │   └── app.py
│   └── streamlit/
├── knowledge_graph_core/
│   ├── data_ingestion/
│   ├── kg_construction/
│   │   ├── kg_creation.py
│   │   ├── kg_utils.py
│   │   └── visualization.py
│   ├── kg_rag/
│   │   ├── inference.py
│   │   ├── kg_config.py
│   │   └── kg_operations.py
│   └── prompts/
├── utils/
├── .env.example
├── requirements.txt
└── README.md
```

- **Knowledge Graph Core**: Manages document ingestion, graph construction, and RAG operations
- **API Layer**: FastAPI-based endpoints for code generation
- **Demo Interfaces**: Gradio and Streamlit UIs for easy interaction
- **Caching Layer**: Redis-based caching for performance optimization

## Deployment Flexibility

The system can be deployed in various ways:

1. **Cloud Providers**:
   - AWS (current setup with S3)
   - Google Cloud (using GCS for storage)
   - Azure (using Blob Storage)
   - Any cloud with object storage

2. **On-Premise**:
   - Private cloud infrastructure
   - Local data centers
   - MinIO for S3-compatible storage

3. **Local Development**:
   - Personal development machines
   - CI/CD environments
   - Testing environments

## Prerequisites

- Python 3.8+
- pip package manager
- Redis (optional, for caching)
- Git (for repository access)

## Installation and Setup

The system uses pre-built knowledge graphs stored in S3. Here's how to set it up:

1. Clone the repository:
   ```bash
   git clone https://github.com/neurons-lab/dApp-codegen.git
   cd dApp-codegen
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file with your configuration:
   ```
   # Required - for accessing knowledge graphs
   AWS_ACCESS_KEY_ID=your_aws_access_key
   AWS_SECRET_ACCESS_KEY=your_aws_secret_key
   AWS_DEFAULT_REGION=your_aws_region
   
   # Optional
   REDIS_URL=your_redis_url  # For caching
   LANGFUSE_PUBLIC_KEY=your_langfuse_key  # For observability
   LANGFUSE_SECRET_KEY=your_langfuse_secret
   ```

The knowledge graphs are already available in S3 and will be automatically downloaded when you run the application. No additional setup is required for the knowledge graphs.

## Knowledge Graph System

### Available Knowledge Graphs

The following pre-built knowledge graphs are available:

- `substrate`: Substrate framework knowledge graph
  - Core concepts and patterns
  - Common implementations
  - Best practices and examples

- `ink`: Ink! smart contract knowledge graph
  - Contract patterns
  - Ink! specific features
  - Integration examples

- `solidity`: Solidity smart contract knowledge graph
  - Smart contract patterns
  - Security considerations
  - Common implementations

- `rust`: Rust programming language knowledge graph
  - Language features
  - Common patterns
  - Standard library usage

These knowledge graphs are automatically downloaded from S3 when needed. The knowledge graphs are designed to be:

- **Portable**: Can be moved between different storage systems
- **Versioned**: Support for different versions of frameworks
- **Extensible**: Can be enhanced with additional knowledge
- **Efficient**: Optimized for quick retrieval and minimal storage

### Knowledge Graph Format

The knowledge graphs use a standardized format that:
- Captures code relationships and context
- Stores semantic information
- Enables efficient querying
- Supports incremental updates

This format makes it easy to:
- Move between different storage systems
- Back up and restore knowledge
- Share between different deployments
- Extend with custom knowledge

## Usage

### Running the API

To start the FastAPI endpoint:

```bash
cd api
uvicorn app:app --host 0.0.0.0 --port 8081
```

### Running the Demos

#### Gradio Demo App:
Navigate to the `demos/gradio` directory and run:
```bash
python app_v2.py
```

#### Streamlit Demo App:
Navigate to the `demos/streamlit` directory and run:
```bash
streamlit run app.py
```

## Example Inference Script

Here's a refactored example script to test the inference:

```python
import requests
from requests.auth import HTTPBasicAuth
import json
import time

def test_inference(api_url, prefix_code, username, password):
    payload = {
        "prefix_code": prefix_code,
        "kg_name": "substrate", #replace with other kg_name if needed e.g: ink
    }

    start_time = time.time()

    try:
        with requests.post(api_url, json=payload, auth=HTTPBasicAuth(username, password), stream=True) as response:
            if response.status_code == 200:
                full_response = ""
                for chunk in response.iter_content(chunk_size=512):
                    if chunk:
                        decoded_chunk = chunk.decode('utf-8')
                        full_response += decoded_chunk
                        print(decoded_chunk, end='', flush=True)
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"\n\nElapsed time: {elapsed_time:.2f} seconds")
                return full_response
            else:
                print(f"Request failed with status code {response.status_code}")
                print(response.text)
                return None
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    API_URL = "http://localhost:8081/v1/generate_code"  # Update with your API URL
    USERNAME = "dapp-user"
    PASSWORD = ""  # Replace with actual password
    
    prefix_code = """///Common Generic traits Definition for pallets \n   pub type AccountOf<T>"""
    
    result = test_inference(API_URL, prefix_code, USERNAME, PASSWORD)
    if result:
        print("\nFull response:")
        print(result)
```

Save this script as `test_inference.py` in your project root and run it with:

```bash
python test_inference.py
```
