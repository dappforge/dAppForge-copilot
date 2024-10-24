# dAppForge Copilot

This repository demonstrates the use of a code generation capabilities for blockchain developers. The application uses a Large Language Model for code completion in the Rust programming language and the Substrate framework. Additionally, it is enhanced with a Knowledge Graph and leverages a RAG architecture for improved performance.

# Architecture 
<img width="7792" alt="dAPP" src="https://github.com/neurons-lab/dAppForge-copilot/assets/5167126/5c3e1ad3-1737-48b7-971a-534a46c54f77">


## Agent Chatbot Flow

![Chatbot Flow](https://raw.githubusercontent.com/dappforge/dAppForge-copilot/main/assets/chatbot_agent.png)


We have also added a new chatbot implementation that handles user queries through two routes:

1. **Code Generation Route:** If the user's query is related to code generation, the chatbot loads the relevant Knowledge Graphs to provide an answer based on code generation.
2. **Generic dApp Plugin Questions:** If the query is generic and about the dApp plugin, the chatbot loads the relevant vector data from the vector store to provide an appropriate answer.

This agentic chatbot implementation improves response accuracy by analyzing the query type and selecting the right data source accordingly.

## Prerequisites

- Python 3.x
- Pip package manager
- `requirements.txt`
- `.env` file with your necessary access keys
- `.pem` keys for SSH access to the EC2 instance

## EC2 Instance Access

To log into the running EC2 instance, use the `.pem` file containing the keys with the following command:

```bash
ssh -i "<path_to_your_pem_file_keys>.pem" ubuntu@xxx-xx-xxx-xxx-xxx.compute-x.amazonaws.com
```

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/neurons-lab/dAppForge-copilot.git
   cd dAppForge-copilot
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up the environment variables:**
   ```bash
   cp .env.example .env
   ```
   Structure your `.env` file as follows:
   ```bash
   WANDB_API_KEY=<your_wandb_key_here>
   AWS_ACCESS_KEY_ID=<aws_access_key_here>
   AWS_SECRET_ACCESS_KEY=<aws_secret_access_key_here>
   GITHUB_TOKEN=<github_token_here>
   ```

## Usage

To start the FastAPI endpoint, run the following commands:

```bash
cd api
uvicorn api.main:app --host 0.0.0.0 --port 8081
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
    API_URL = "http://23.20.247.78:8081/v1/generate_code"  # Update with your API URL
    USERNAME = "dapp-user"
    PASSWORD = "  # Replace with actual password
    
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




## Directory Descriptions
```
├── api : Contains the FastAPI endpoint files.
│   ├── __init__.py
│   ├── app.py
│   ├── auth.py
│   ├── hash_passwords.py
│   ├── README.md
│   ├── users.yaml
│   └── utils.py
├── assets : Contains image assets.
│   └── chatbot_agent.png
├── caching : Contains caching-related files.
│   ├── __init__.py
│   └── redis_cache.py
├── knowledge_graph_core : Core knowledge graph functionality.
│   ├── data_ingestion
│   │   ├── github_connector.py
│   │   └── website_connector.py
│   ├── kg_construction
│   │   ├── kg_creation.py
│   │   ├── kg_utils.py
│   │   └── visualization.py
│   ├── kg_rag
│   │   ├── inference.py
│   │   ├── kg_config.py
│   │   └── kg_operations.py
│   └── prompts
│       ├── chat_conversation.py
│       ├── code_completion.py
│       ├── kg_triplets.py
│       ├── text_qa_template.py
│       └── __init__.py
├── services : Contains service-related files.
│   ├── fastapi_app.service
│   ├── gradio_app.service
│   ├── README.md
│   └── streamlit.service
├── utils : Utility files and configurations.
│   ├── __init__.py
│   ├── config.json
│   ├── config.py
│   ├── load_and_persist_kg.py
│   ├── models.py
│   └── utils.py
├── .env.example : Example environment file.
├── LICENSE : License file.
├── README.md : Main README file.
└── requirements.txt : List of Python dependencies.
```
