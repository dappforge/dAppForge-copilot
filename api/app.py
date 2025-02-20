import os
import sys
import json
import bcrypt
import logging
import httpx
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import StreamingResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.utils import check_and_trim_code_length, prepare_response, extract_value_from_generated_code, clean_generated_code
from api.auth import load_users_from_yaml
from utils.config import start_wandb_run
from utils.models import CodeRequest, CodeResponse, KGCreationRequest, MergeKGRequest, ChatRequest
from utils.load_and_persist_kg import load_and_persist_kg
from knowledge_graph_core.data_ingestion.github_connector import load_github_documents
from knowledge_graph_core.data_ingestion.website_connector import scrape_website
from knowledge_graph_core.kg_construction.kg_creation import create_knowledge_graph_index, create_kg_triplet_extraction_template
from knowledge_graph_core.kg_construction.kg_utils import set_llms, load_environment_variables, persist_knowledge_graph, detect_source, extract_owner_repo
from knowledge_graph_core.kg_construction.visualization import visualize_knowledge_graph
from knowledge_graph_core.kg_rag.inference import claude_inference, claude_inference_neptune, composable_graph_inference, claude_inference_streaming, claude_chat_streaming
from knowledge_graph_core.kg_rag.kg_operations import load_kg_index
from caching.redis_cache import generate_cache_key, get_cached_result, set_cache_result
from caching.redis_cache import async_generate_cache_key, async_get_cached_result, async_set_cache_result
from feedback import FeedbackRequest, FeedbackResponse
from feedback.handler import FeedbackHandler

from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import StorageContext, SummaryIndex
from llama_index.core.indices.composability import ComposableGraph


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


feedback_handler = FeedbackHandler()


# Load users from YAML file
users = load_users_from_yaml('users.yaml')

PERSIST_DIR = '/home/ubuntu/dApp/knowledge_graph_data/'


def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    hashed_password = users.get(credentials.username)
    if not hashed_password or not bcrypt.checkpw(credentials.password.encode('utf-8'), hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.post("/v1/create_kg")
async def create_kg(request: KGCreationRequest, username: str = Depends(authenticate)):
    """
    Create a Knowledge Graph (KG) from provided URLs.
    
    Args:
        request (KGCreationRequest): Contains URLs and optional KG name.
        username (str): Authenticated username, injected by the `authenticate` dependency.
    
    Returns:
        dict: A message confirming the creation of the Knowledge Graph.
    """
    if not request.urls:
        raise HTTPException(status_code=400, detail="No URLs provided")

    documents = []

    for url in request.urls:
        source = detect_source(url)
        if source == "github":
            openai_api_key, github_token = load_environment_variables()
            set_llms()
            owner, repo = extract_owner_repo(url)
            if owner and repo:
                gh_docs = load_github_documents(github_token, owner, repo)
                if gh_docs:
                    documents.extend(gh_docs)
        else:
            set_llms()
            all_docs = scrape_website(url)
            if all_docs:
                documents.extend(all_docs)

    if documents:
        triplet_template = create_kg_triplet_extraction_template(KG_TRIPLETS_TEMPLATE)
        index = create_knowledge_graph_index(documents, triplet_template)
        if index is None:
            raise HTTPException(status_code=500, detail="Failed to create Knowledge Graph Index")
        visualize_knowledge_graph(index, f"{request.kg_name}_dump.html")
        persist_knowledge_graph(index, request.kg_name, request.urls)
        return {"message": "Knowledge Graph created"}

    raise HTTPException(status_code=400, detail="No valid documents found")

@app.post("/v1/merge_kg")
async def merge_kg(request: MergeKGRequest, username: str = Depends(authenticate)):
    """
    Merge two or more existing Knowledge Graphs (KGs) from S3 based on their names and persist the merged KG to S3.
    
    Args:
        request (MergeKGRequest): Request containing the names of the KGs to be merged and the prefix code for the merged KG.
        username (str): The username of the authenticated user (injected by the `authenticate` dependency).
    
    Returns:
        CodeResponse: A response containing the generated code, knowledge graph edges, and subgraph plot.
    """
    nest_asyncio.apply()

    try:
        # Load and persist knowledge graphs
        load_and_persist_kg(request.kg_names)

        # Load the knowledge graph indexes
        kgs = []
        for kg_name in request.kg_names:
            persist_path = os.path.join(PERSIST_DIR, kg_name.replace('/', '_'))
            kg = load_kg_index_from_disk(persist_path)
            kgs.append(kg)

        # Merge the knowledge graphs
        graph_store = SimpleGraphStore()
        storage_context = StorageContext.from_defaults(graph_store=graph_store)
        merged_graph = ComposableGraph.from_indices(
            SummaryIndex,
            kgs,
            index_summaries=[f"Summary of {kg}" for kg in request.kg_names],
            storage_context=storage_context,
        )

        # Cache result and return response
        cache_key = generate_cache_key(request.prefix_code)
        cached_result = get_cached_result(cache_key)
        if cached_result:
            return CodeResponse(**cached_result)

        generated_code, sub_edges, subplot = composable_graph_inference(merged_graph, request.prefix_code)
        response = prepare_response(generated_code, sub_edges, subplot)
        set_cache_result(cache_key, response.dict())

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

    # Use the asynchronous version of generate_cache_key
    cache_key = await async_generate_cache_key(prefix_code)
    
    # Use the asynchronous version of get_cached_result
    cached_result = await async_get_cached_result(cache_key)

    if cached_result:
        return CodeResponse(**cached_result)

    generated_code, sub_edges, subplot = claude_inference(prefix_code, request.kg_name)
    response = prepare_response(generated_code, sub_edges, subplot)

    # Use the asynchronous version of set_cache_result
    await async_set_cache_result(cache_key, response.dict())

    return response





@app.post("/v1/generate_code_neptune", response_model=CodeResponse)
async def generate_code_neptune(request: CodeRequest, username: str = Depends(authenticate)):
    """
    Generates code based on a defined prefix and returns the generated code along with knowledge graph edges and a subgraph plot.
    
    Args:
        request (CodeRequest): Request containing the prefix code for code generation.
        username (str): Authenticated username, provided by the dependency injection.
    
    Returns:
        CodeResponse: A response model containing the generated code, knowledge graph edges, and a subgraph plot.
    """
    prefix_code = check_and_trim_code_length(request.prefix_code)

    # Use the asynchronous version of generate_cache_key
    cache_key = await async_generate_cache_key(prefix_code)
    
    # Use the asynchronous version of get_cached_result
    cached_result = await async_get_cached_result(cache_key)

    if cached_result:
        return CodeResponse(**cached_result)

    generated_code, sub_edges, subplot = claude_inference_neptune(prefix_code, request.kg_name)
    response = prepare_response(generated_code, sub_edges, subplot)

    # Use the asynchronous version of set_cache_result
    await async_set_cache_result(cache_key, response.dict())

    return response

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
        media_type="text/event-stream"
    )

@app.post("/v1/chat_code")
async def generate_code(request: ChatRequest, username: str = Depends(authenticate)):
    """
    Generates code based on a defined prefix in a chat-based format and streams the generated code as it is created.
    
    Args:
        request (ChatRequest): Request containing the query, kg_name, and session_id for chat generation.
        username (str): Authenticated username, provided by the dependency injection.
    
    Returns:
        StreamingResponse: A streaming response that outputs the generated chat in real-time, with a media type of "text/event-stream".
    """
    return StreamingResponse(
        claude_chat_streaming(request.query, request.kg_name, request.session_id),
        media_type="text/event-stream"
    )


@app.post("/v1/feedback/{user_id}/{session_id}", response_model=FeedbackResponse)
async def submit_feedback(
    user_id: str,
    session_id: str,
    request: FeedbackRequest,
    username: str = Depends(authenticate)
):
    """
    Submit user feedback for a session. Feedback is saved asynchronously and rules
    are processed in the background.
    """
    try:
        feedback_data = request.dict()
        feedback_data["user_id"] = user_id
        feedback_data["session_id"] = session_id
        
        result = await feedback_handler.save_feedback(feedback_data)
        return FeedbackResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    uvicorn.run(app, host="0.0.0.0", port=8081, loop="asyncio", workers=4, timeout_keep_alive=180)
