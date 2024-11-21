from datetime import datetime
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from utils.utils import plot_subgraph_via_edges
from .kg_config import template, stream_template, chat_template
from .kg_operations import create_query_engine, create_streaming_query_engine, kg_index_substrate, kg_index_neptune, kg_index_ink, load_kg_index, vector_index, kg_index_substrate_tool, kg_index_ink_tool, vector_tool
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
    query = template.format(prefix_code=prefix_code)



    if kg_name == "substrate":
        index = kg_index_substrate
    elif kg_name == "ink":
        index = kg_index_ink
    else:
        raise ValueError(f"Unknown kg_name: {kg_name}")

    print(f"Using index for {kg_name}")
    print(f"Current embed_model: {Settings.embed_model}")
    
    # Log the dimensions of the query embedding
    query_embedding = Settings.embed_model.get_text_embedding(query)
    print(f"Query embedding dimensions: {len(query_embedding)}")

    # Create a new query engine with the current settings
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
    
    # Step 1: Generate cache key based on kg_name and query
    cache_key = await async_generate_cache_key(kg_name, query)
    cached_result = await async_get_cached_result(cache_key)
    
    if cached_result:
        for token in cached_result:
            yield token
        return

    # Step 4: Proceed with usual inference if no cached result
    chat_store_filename = f"{session_id}_{datetime.now().strftime('%Y%m%d')}.json"
    try:
        chat_store = SimpleChatStore.from_persist_path(persist_path=chat_store_filename)
    except FileNotFoundError:
        chat_store = SimpleChatStore()

    chat_memory = ChatMemoryBuffer.from_defaults(
        token_limit=16000,
        chat_store=chat_store,
        chat_store_key=session_id,
    )
    
    # Prepare indices for router
    indices = {
        'substrate': kg_index_substrate,
        'ink': kg_index_ink,
        'vector': vector_index
    }
    
    # Use new router chat streaming
    result_tokens = []
    async for token in router_chat_streaming(
        query=query,
        kg_name=kg_name,
        session_id=session_id,
        chat_memory=chat_memory,
        indices=indices
    ):
        result_tokens.append(token)
        yield token

    # Cache results
    await async_set_cache_result(cache_key, result_tokens)
    chat_store.persist(persist_path=chat_store_filename)






# async def claude_chat_streaming(query, kg_name, session_id, suffix="}"):
#     """Perform chat inferencing using Claude with streaming response and caching."""
    
#     # Step 1: Generate cache key based on kg_name and query
#     cache_key = await async_generate_cache_key(kg_name, query)
    
#     # Step 2: Check if there is a cached result for this key
#     cached_result = await async_get_cached_result(cache_key)
    
#     if cached_result:
#         # Step 3: If cached result exists, return it
#         for token in cached_result:
#             yield token
#         return

#     # Step 4: Proceed with usual inference if no cached result
#     chat_store_filename = f"{session_id}_{datetime.now().strftime('%Y%m%d')}.json"
#     try:
#         chat_store = SimpleChatStore.from_persist_path(persist_path=chat_store_filename)
#     except FileNotFoundError:
#         chat_store = SimpleChatStore()

#     chat_memory = ChatMemoryBuffer.from_defaults(
#         token_limit=16000,
#         chat_store=chat_store,
#         chat_store_key=session_id,
#     )
    
#     index = kg_index_substrate if kg_name == "substrate" else kg_index_ink
#     custom_retriever = index.as_retriever(similarity_top_k=5)
    
#     retriever = RouterRetriever(
#         selector=LLMSingleSelector.from_defaults(),
#         retriever_tools=[
#             RetrieverTool(
#                 retriever=custom_retriever,
#                 metadata=ToolMetadata(
#                     name="custom_kg_retriever",
#                     description="Custom knowledge graph retriever"
#                 )
#             ),
#             vector_tool,
#         ],
#     )
    
#     formatted_query = chat_template.format(query=query)
#     context_chat_engine = ContextChatEngine.from_defaults(
#         retriever=retriever,
#         memory=chat_memory
#     )

#     # Step 5: Capture streaming response and cache it
#     result_tokens = []
#     streaming_response = context_chat_engine.stream_chat(formatted_query)
#     for token in streaming_response.response_gen:
#         result_tokens.append(token)
#         yield token  # Yield each token in real-time

#     # Step 6: Cache the result after streaming completes
#     await async_set_cache_result(cache_key, result_tokens)

#     # Persist chat store
#     chat_store.persist(persist_path=chat_store_filename)

async def claude_inference_streaming(prefix_code, kg_name, suffix="}"):
    """Perform inference using Claude with streaming response."""
    query = stream_template.format(prefix_code=prefix_code)
    
    if kg_name == "substrate":
        streaming_query_engine = create_streaming_query_engine(kg_index_substrate)
    elif kg_name == "ink":
        streaming_query_engine = create_streaming_query_engine(kg_index_ink)
        
    streaming_response = streaming_query_engine.query(query)
    for token in streaming_response.response_gen:
        yield token


