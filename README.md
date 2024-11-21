# dApp Code Generation

This repository demonstrates a code generation app, served as both an API endpoint and a Gradio demo app. The application uses a Large Language Model for code completion in Rust Programming language and Substrate framework, enhanced with a Knowledge Graph and leveraging a RAG system for improved performance.

## Prerequisites

- Python 3.x
- Pip package manager
- `.env` file with necessary access keys
- `.pem` keys for SSH access to the EC2 instance (if applicable)

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

## Setup and Installation

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
   Edit the `.env` file with your specific keys and settings.

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
