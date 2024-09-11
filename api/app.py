import os
import sys
import json
import bcrypt
import yaml
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import status
import logging
from llama_index.core import StorageContext, KnowledgeGraphIndex, Settings
from llama_index.readers.web import WholeSiteReader
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import StorageContext, SummaryIndex
import s3fs
from llama_index.readers.web import WholeSiteReader
from llama_index.core import SummaryIndex
from llama_index.core.indices.composability import ComposableGraph
from llama_index.core import StorageContext, load_index_from_storage
import nest_asyncio
import re
import json
import httpx
from fastapi.responses import StreamingResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.utils import check_and_trim_code_length, prepare_response, clean_and_escape_code_logic2, extract_value_from_generated_code, clean_generated_code, load_kg_index_from_disk, process_generated_code, detect_source, load_users_from_yaml
from common.config import start_wandb_run
from common.models import CodeRequest, CodeResponse, KGCreationRequest, MergeKGRequest
from common.inference import claude_inference, composable_graph_inference, load_kg_index, plot_full_kg, claude_inference_streaming
from common.utils import extract_code_from_response, extract_code_using_regex
from caching.redis_cache import generate_cache_key, get_cached_result, set_cache_result, invalidate_cache
from code_generation.kg_construction.load_and_persist_kg import load_and_persist_kg
from code_generation.kg_construction.website_documents_creation import scrape_website

import asyncio
from concurrent.futures import ProcessPoolExecutor
PERSIST_DIR = '/home/ubuntu/dApp/knowledge_graph_data/'

app = FastAPI()
start_wandb_run()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBasic()

# Load users from YAML file
users = load_users_from_yaml('users.yaml')


# Initialize S3 filesystem
s3 = s3fs.S3FileSystem(anon=False)

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    hashed_password = users.get(credentials.username)
    if not hashed_password or not bcrypt.checkpw(credentials.password.encode('utf-8'), hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

async def async_generate_cache_key(prefix_code: str) -> str:
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor() as pool:
        return await loop.run_in_executor(pool, generate_cache_key, prefix_code)

async def async_get_cached_result(cache_key: str):
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor() as pool:
        return await loop.run_in_executor(pool, get_cached_result, cache_key)

async def async_set_cache_result(cache_key: str, result: dict):
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor() as pool:
        await loop.run_in_executor(pool, set_cache_result, cache_key, result)


@app.post("/v1/generate_stream_code")
async def generate_code(request: CodeRequest, username: str = Depends(authenticate)):
    """
    Generates code based on a defined prefix and streams the generated code as it is created.

    Args:
        request (CodeRequest): Request containing the prefix code for code generation.
        username (str): Authenticated username, provided by the dependency injection.

    Returns:
        StreamingResponse: A streaming response that outputs the generated code in real-time, with a media type of "text/event-stream".
    """
    return StreamingResponse(
        claude_inference_streaming(request.prefix_code, request.kg_name),
        media_type="text/event-stream")


@app.post("/v1/generate_code", response_model=CodeResponse)
async def generate_code(request: CodeRequest, username: str = Depends(authenticate)):
    """
    Generates code based on a defined prefix and returns the generated code along with knowledge graph edges and a subgraph plot.

    Args:
        request (CodeRequest): Request containing the prefix code for code generation.
        username (str): Authenticated username, provided by the dependency injection.

    Returns:
        CodeResponse: A response model containing the generated code, knowledge graph edges, and a subgraph plot.
    """
    prefix_code = check_and_trim_code_length(request.prefix_code)
               
    cache_key = await async_generate_cache_key(prefix_code)
    cached_result = await async_get_cached_result(cache_key)
    
    if cached_result:
        return CodeResponse(**cached_result)

    generated_code, sub_edges, subplot = claude_inference(prefix_code,request.kg_name)    
    response = prepare_response(generated_code, sub_edges, subplot)
    
    await async_set_cache_result(cache_key, response.dict())
    
    return response




@app.get("/")
async def root():
    """
    Root endpoint that welcomes users to the API.

    Returns:
        dict: A welcome message.
    """
    return {"message": "Welcome to the dApp KG+LLM API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081, loop="asyncio", workers = 4,  timeout_keep_alive=180)