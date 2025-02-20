from datetime import datetime
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from utils.utils import plot_subgraph_via_edges
from .kg_config import template, stream_template, framework_prompts
from .kg_operations import create_query_engine, create_streaming_query_engine, kg_index_substrate,kg_index_solidity, kg_index_neptune, kg_index_ink,kg_index_rust, load_kg_index, vector_index, kg_index_substrate_tool, kg_index_ink_tool, vector_tool
from llama_index.core.tools import RetrieverTool
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.selectors import (
    PydanticMultiSelector,
    PydanticSingleSelector,
)
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.tools import ToolMetadata
from llama_index.core import Settings
from caching.redis_cache import async_generate_cache_key, async_get_cached_result, async_set_cache_result
from .kg_router import router_chat_streaming
import os
from pathlib import Path



def load_feedback_rules() -> str:
    """Load feedback rules from JSON file."""
    try:
        # Get base directory and construct path to rules file
        base_dir = Path(__file__).parent.parent
        rules_file = base_dir / 'feedback' / 'rules' / 'feedback_rules.json'
        
        if rules_file.exists():
            with open(rules_file, 'r') as f:
                rules_data = json.load(f)
                rules = rules_data.get("rules", [])
                if rules:
                    rules_section = "\nUser Feedback-Based Rules:\n"
                    rules_section += "\n".join(rules)
                    return rules_section
        return ""
    except Exception as e:
        print(f"Error loading feedback rules: {e}")
        return ""


def composable_graph_inference(composable_graph, prefix_code):
    """Perform inference based on multiple knowledge graphs"""
    graph_query_engine = composable_graph.as_query_engine(
        include_text=True,
        response_mode='refine',
        graph_store_query_depth=3,
        similarity_top_k=5,
        use_gpu=True)

    query = template.format(prefix_code=prefix_code)
    response = graph_query_engine.query(query)
    sub_edges, subplot = plot_subgraph_via_edges(response.metadata)

    return response.response, sub_edges, subplot

def claude_inference(prefix_code, kg_name, suffix="}"):
    """Perform inference using Claude and return the generated code, edges, and subplot."""
    # Load and append feedback rules to the template
    rules_section = load_feedback_rules()
    query = template.format(prefix_code=prefix_code)
    if rules_section:
        query = f"{query}\n{rules_section}"

    if kg_name == "substrate":
        index = kg_index_substrate
    elif kg_name == "ink":
        index = kg_index_ink
    elif kg_name == "solidity":
        index = kg_index_solidity
    elif kg_name == "rust":
        index = kg_index_rust
    else:
        raise ValueError(f"Unknown kg_name: {kg_name}")

    print(f"Using index for {kg_name}")
    print(f"Current embed_model: {Settings.embed_model}")
    
    query_embedding = Settings.embed_model.get_text_embedding(query)
    print(f"Query embedding dimensions: {len(query_embedding)}")

    query_engine = create_query_engine(index)
    response = query_engine.query(query)
    return response.response, [], ""


def claude_inference_neptune(prefix_code, kg_name, suffix="}"):
    
    """Perform inference using Claude and return the generated code, edges, and subplot."""
    query = template.format(prefix_code=prefix_code)

    # if kg_name == "substrate":
    #     query_engine = create_query_engine(kg_index_neptune)
    # elif kg_name == "ink":
    #     query_engine = create_query_engine(kg_index_neptune)
    query_engine = create_query_engine(kg_index_neptune)
    response = query_engine.query(query)
    return response.response, [], ""

def claude_inference_gradio(prefix_code, suffix="}"):
    """Perform inference using Claude for Gradio interface."""
    query = template.format(prefix_code=prefix_code)
    query_engine = create_query_engine(kg_index_ink)
    response = query_engine.query(query)
    sub_edges, subplot = plot_subgraph_via_edges(response.metadata)
    return response.response, sub_edges, subplot


async def claude_chat_streaming(query, kg_name, session_id, suffix="}"):
    """Perform chat inferencing using Claude with streaming response and caching."""
    # Load feedback rules
    rules_section = load_feedback_rules()
    
    # Format query with framework-specific prompt and feedback rules
    prompt_template = framework_prompts.get(kg_name, framework_prompts['base'])
    formatted_query = prompt_template.format(query=query)
    if rules_section:
        formatted_query = f"{formatted_query}\n{rules_section}"
    
    # Cache key generation should include rules to ensure proper caching
    cache_key = await async_generate_cache_key(kg_name, formatted_query)
    cached_result = await async_get_cached_result(cache_key)
    
    if cached_result:
        for token in cached_result:
            yield token
        return

    # Rest of the function remains the same
    session_data_dir = os.path.join(os.path.dirname(__file__), 'session_data')
    os.makedirs(session_data_dir, exist_ok=True)
    
    chat_store_filename = os.path.join(
        session_data_dir,
        f"{session_id}_{datetime.now().strftime('%Y%m%d')}.json"
    )
    
    try:
        chat_store = SimpleChatStore.from_persist_path(persist_path=chat_store_filename)
    except FileNotFoundError:
        chat_store = SimpleChatStore()

    chat_memory = ChatMemoryBuffer.from_defaults(
        token_limit=16000,
        chat_store=chat_store,
        chat_store_key=session_id,
    )
    
    indices = {
        'substrate': kg_index_substrate,
        'ink': kg_index_ink,
        'solidity': kg_index_solidity,
        'rust': kg_index_rust,
        'vector': vector_index
    }
    
    result_tokens = []
    async for token in router_chat_streaming(
        query=formatted_query,
        kg_name=kg_name,
        session_id=session_id,
        chat_memory=chat_memory,
        indices=indices
    ):
        result_tokens.append(token)
        yield token

    await async_set_cache_result(cache_key, result_tokens)
    chat_store.persist(persist_path=chat_store_filename)


async def claude_inference_streaming(prefix_code, kg_name, suffix="}"):
    """Perform inference using Claude with streaming response."""
    query = stream_template.format(prefix_code=prefix_code)
    
    if kg_name == "substrate":
        streaming_query_engine = create_streaming_query_engine(kg_index_substrate)
    elif kg_name == "ink":
        streaming_query_engine = create_streaming_query_engine(kg_index_ink)
    elif kg_name == "solidity":
        streaming_query_engine = create_streaming_query_engine(kg_index_solidity)
    elif kg_name == "rust":
        streaming_query_engine = create_streaming_query_engine(kg_index_rust)
        
    streaming_response = streaming_query_engine.query(query)
    for token in streaming_response.response_gen:
        yield token


