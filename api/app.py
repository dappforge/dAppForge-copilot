import os
import sys
import json
import bcrypt
import logging
import httpx
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import StreamingResponse, Response
from pathlib import Path
import uuid

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from api.utils import check_and_trim_code_length, prepare_response, extract_value_from_generated_code, clean_generated_code
from api.auth import load_users_from_yaml, authenticate
from utils.config import configure_settings
from utils.models import CodeRequest, CodeResponse, ChatRequest

 
from knowledge_graph_core.kg_rag.inference import claude_inference, composable_graph_inference, claude_inference_streaming, claude_chat_streaming, get_memgraph_query_engine, warmup_all_engines
from knowledge_graph_core.kg_rag.inference import MEMGRAPH_CONFIGS

from caching.redis_cache import generate_cache_key, get_cached_result, set_cache_result
from caching.redis_cache import async_generate_cache_key, async_get_cached_result, async_set_cache_result
from feedback import FeedbackRequest, FeedbackResponse
from feedback.handler import FeedbackHandler

from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import StorageContext, SummaryIndex
from llama_index.core.indices.composability import ComposableGraph
from api.langfuse_utils import langfuse_handlers, flush_langfuse
from utils.ddb_utils import DdbTable, Session as DdbSession, Message as DdbMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure settings
Settings = configure_settings()

security = HTTPBasic()

feedback_handler = FeedbackHandler()

# Load users from YAML file
users_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'users.yaml')
users = load_users_from_yaml(users_file_path)


def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    hashed_password = users.get(credentials.username)
    if not hashed_password or not bcrypt.checkpw(credentials.password.encode('utf-8'), hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

 



@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    if langfuse_handlers:
        # Set the callback manager in Settings
        Settings.callback_manager = langfuse_handlers['callback_manager']
        logger.info("Langfuse handlers initialized successfully")
        try:
            # Attach callback manager to the Bedrock LLM instance used by KG RAG
            from knowledge_graph_core.kg_rag.kg_config import bedrock_llm as _bedrock_llm
            _bedrock_llm.callback_manager = Settings.callback_manager
            logger.info("Attached Langfuse callback manager to Bedrock LLM")
            logger.info(f"Bedrock LLM model: {getattr(_bedrock_llm, 'model', 'unknown')}")
        except Exception as e:
            logger.error(f"Failed to attach callback manager to Bedrock LLM: {e}")
    # Preload Memgraph drivers and query engines for all configured KGs
    try:
        if isinstance(MEMGRAPH_CONFIGS, dict):
            logger.info("Warming up drivers and query engines for all KGs...")
            warmup_all_engines()
            logger.info("Warmup complete.")
    except Exception as e:
        logger.error(f"Engine preloading encountered an error: {type(e).__name__}: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if langfuse_handlers:
        try:
            flush_langfuse()
            logger.info("Langfuse events flushed successfully")
        except Exception as e:
            logger.error(f"Error flushing Langfuse events: {str(e)}")

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

    generated_code, sub_edges, subplot, input_tokens, output_tokens = claude_inference(prefix_code, request.kg_name)

    # Best-effort: persist this generate_code interaction to DynamoDB with a UUID
    try:
        request_id = str(uuid.uuid4())
        import os
        table_name = os.getenv("DDB_CHAT_TABLE", "DAppChatSessionsTable")
        region = os.getenv("AWS_REGION", "us-east-1")
        ddb = DdbTable(table_name, region)
        user_msg = DdbMessage(role="user", content=f"prefix_code:\n{prefix_code}", additional_kwargs={"kg_name": request.kg_name})
        assistant_msg = DdbMessage(role="assistant", content=generated_code, additional_kwargs={})
        sess = DdbSession(id=request_id, messages=[user_msg, assistant_msg], metadata={"kg_name": request.kg_name, "endpoint": "generate_code"})
        ddb.create_session(sess)
    except Exception:
        pass

    response = prepare_response(generated_code, sub_edges, subplot, input_tokens, output_tokens)
    await async_set_cache_result(cache_key, response)
    return response



@app.post("/v1/generate_stream_code")
async def generate_code(request: CodeRequest, username: str = Depends(authenticate)):
    """Stream the code generation response"""
    prefix_code = check_and_trim_code_length(request.prefix_code)
    return StreamingResponse(
        claude_inference_streaming(prefix_code, request.kg_name),
        media_type="text/event-stream"
    )

@app.post("/v1/chat_code")
async def chat_code(request: ChatRequest, username: str = Depends(authenticate)):
    """Stream the chat response"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        return StreamingResponse(
            claude_chat_streaming(
                query=request.query,
                kg_name=request.kg_name,
                session_id=session_id
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Error in chat_code endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/chat_tokens/{session_id}")
async def get_chat_tokens(session_id: str, username: str = Depends(authenticate)):
    """Get token counts for the session.
    Returns last-turn and cumulative totals if available.
    """
    try:
        from knowledge_graph_core.kg_rag.inference import token_storage
        
        # Prefer persisted counts from DynamoDB if available
        try:
            table_name = os.getenv("DDB_CHAT_TABLE", "DAppChatSessionsTable")
            region = os.getenv("AWS_REGION", "us-east-1")
            ddb = DdbTable(table_name, region)
            session = ddb.get_session(session_id)
            if session and getattr(session, "metadata", None):
                meta = session.metadata or {}
                if ("last_input_tokens" in meta) or ("last_output_tokens" in meta) or ("total_input_tokens" in meta) or ("total_output_tokens" in meta):
                    return {
                        "input_tokens": int(meta.get("last_input_tokens", 0)),
                        "output_tokens": int(meta.get("last_output_tokens", 0)),
                        "total_input_tokens": int(meta.get("total_input_tokens", 0)),
                        "total_output_tokens": int(meta.get("total_output_tokens", 0)),
                        "turn_count": int(meta.get("turn_count", 0)),
                    }
        except Exception:
            # If DDB is unavailable or any error occurs, fall back to in-memory tokens
            pass

        # Fallback: in-memory token storage
        if session_id in token_storage:
            tokens = token_storage[session_id]
            return {
                "input_tokens": tokens.get('last_input_tokens', 0),
                "output_tokens": tokens.get('last_output_tokens', 0)
            }
        else:
            return {"input_tokens": 0, "output_tokens": 0}
    except Exception as e:
        logger.error(f"Error getting chat tokens: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
    """Root endpoint that welcomes users to the API."""
    return {"message": "Welcome to the dApp KG+LLM API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)