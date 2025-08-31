from llama_index.core import StorageContext, load_index_from_storage, SimpleDirectoryReader, PropertyGraphIndex
from llama_index.core import PromptTemplate
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from .kg_config import fs, logger, S3_PATH_INK, S3_PATH_SUBSTRATE, S3_PATH_SOLIDITY, S3_PATH_RUST, PERSIST_DISK_PATH, text_qa_template_str
from llama_index.core.tools import RetrieverTool
from botocore.exceptions import ClientError
from llama_index.core import ServiceContext
from llama_index.graph_stores.neptune import (
    NeptuneAnalyticsPropertyGraphStore,
    NeptuneDatabasePropertyGraphStore,
)
import nest_asyncio
from llama_index.core import Settings
import os

# Define local storage path
LOCAL_KG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'local_kg_storage')

def get_local_path_for_kg(kg_name: str) -> str:
    """Get the local storage path for a given KG name."""
    return os.path.join(LOCAL_KG_DIR, kg_name)

def load_kg_index_from_s3(s3_path, fs):
    """Load a knowledge graph index from S3."""
    logger.info(f"Loading knowledge graph index from S3: {s3_path}")
    storage_context = StorageContext.from_defaults(persist_dir=s3_path, fs=fs)
    return load_index_from_storage(storage_context)

def load_kg_index_from_local(kg_name: str):
    """Load a knowledge graph index from local storage."""
    
    # TODO: TEMPORARY - Comment out original implementation
    local_path = get_local_path_for_kg(kg_name)
    logger.info(f"Loading knowledge graph index from local storage: {local_path}")
    storage_context = StorageContext.from_defaults(persist_dir=local_path)
    return load_index_from_storage(storage_context)
    
    # TEMPORARY: Always load solidity KG from pickle file regardless of kg_name parameter
    # import pickle
    # from pathlib import Path
    
    # # Define the pickle cache file path (same as solidity comparison script)
    # cache_file = Path("/home/ubuntu/dApp-merx/dApp-codegen/llm-as-a-judge/solidity/solidity_query_engine_cache.pkl")
    
    # if cache_file.exists():
    #     try:
    #         logger.info(f"Loading knowledge graph index from pickle cache: {cache_file}")
    #         with open(cache_file, 'rb') as f:
    #             cached_data = pickle.load(f)
    #             # Return the index from the cached data
    #             return cached_data['index']
    #     except Exception as e:
    #         logger.error(f"Error loading from pickle cache: {str(e)}")
    #         raise
    # else:
    #     logger.error(f"Pickle cache file not found: {cache_file}")
    #     raise FileNotFoundError(f"Pickle cache file not found: {cache_file}")

def load_kg_index(kg_name: str, use_s3: bool = False):
    """Load a knowledge graph index, preferring local storage unless use_s3 is True."""
    if not use_s3:
        local_path = get_local_path_for_kg(kg_name)
        if os.path.exists(local_path):
            return load_kg_index_from_local(kg_name)
        else:
            logger.warning(f"Local KG not found for {kg_name}, falling back to S3.")
    
    s3_paths = {
        'ink': S3_PATH_INK,
        'substrate': S3_PATH_SUBSTRATE,
        'solidity': S3_PATH_SOLIDITY,
        'rust': S3_PATH_RUST
    }
    s3_path = s3_paths.get(kg_name)
    if not s3_path:
        raise ValueError(f"Unknown KG type: {kg_name}")
    
    return load_kg_index_from_s3(s3_path, fs)

def create_query_engine(kg_index, graph_store_query_depth=1, similarity_top_k=3):
    """Create and configure the query engine."""
    logger.info("Creating and configuring the query engine...")
    return kg_index.as_query_engine(
        include_text=True,
        similarity_top_k=similarity_top_k
    )

def create_streaming_query_engine(kg_index, graph_store_query_depth=1, similarity_top_k=3):
    """Create and configure the streaming query engine."""
    logger.info("Creating and configuring the streaming query engine...")
    text_qa_template = PromptTemplate(text_qa_template_str)

    return kg_index.as_query_engine(
        include_text=True,
        response_mode="refine",
        embedding_mode="hybrid",
        graph_store_query_depth=graph_store_query_depth,
        similarity_top_k=similarity_top_k,
        use_gpu=True,
        text_qa_template=text_qa_template,
        streaming=True
    )

def load_kg_index_from_neptune():
    nest_asyncio.apply()
    # Neptune Database connection 
    graph_store = NeptuneDatabasePropertyGraphStore(
        host='db-neptune-1-instance-1.ctoseyq84zjc.us-east-1.neptune.amazonaws.com'
    )

    # Creating the storage context
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store
    )
    logger.info("Loaded kg index from AWS Neptune")

    return index

def create_vector_embeddings():
    try:
        dapp_content_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vector_data", "dapp_content.txt")
        documents = SimpleDirectoryReader(input_files=[dapp_content_path]).load_data()
        logger.info("Loaded how-to document...")
        
        splitter = SentenceSplitter(chunk_size=1024)
        nodes = splitter.get_nodes_from_documents(documents)
        logger.info(f"Created {len(nodes)} nodes from documents")
        
        try:
            vector_index = VectorStoreIndex(nodes)
            return vector_index
        except ClientError as e:
            logger.error(f"Error creating vector index: {str(e)}")
            if e.response['Error']['Code'] == 'ValidationException':
                logger.error("Bedrock service returned a ValidationException. Check your model configuration and parameters.")
            raise
    except Exception as e:
        logger.error(f"Unexpected error in create_vector_embeddings: {str(e)}")
        raise

# Load knowledge graph indices from local storage (falls back to S3 if not found locally)
# kg_index_substrate = load_kg_index('substrate')
# kg_index_ink = load_kg_index('ink')
# kg_index_neptune = load_kg_index_from_neptune()
# kg_index_solidity = load_kg_index('solidity')
# kg_index_rust = load_kg_index('rust')

# Load guide document for dApp co-pilot
vector_index = create_vector_embeddings()

#vector_query_engine = vector_index.as_query_engine()

# kg_index_substrate_retriever = kg_index_substrate.as_retriever()
# kg_index_ink_retriever = kg_index_ink.as_retriever()
# kg_index_solidity_retriever = kg_index_solidity.as_retriever()
# kg_index_rust_retriever = kg_index_rust.as_retriever()


vector_index_retriever = vector_index.as_retriever()


# kg_index_substrate_tool = RetrieverTool.from_defaults(
#     retriever=kg_index_substrate_retriever,
#     description=(
#         "Useful for retrieving context data related to Substrate knowledge graph."
#     ),
# )

# kg_index_ink_tool = RetrieverTool.from_defaults(
#     retriever=kg_index_ink_retriever,
#     description=(
#         "Useful for retrieving specific  data related to Ink knowledge graph."
#     ),
# )

# kg_index_solidity_tool = RetrieverTool.from_defaults(
#     retriever=kg_index_solidity_retriever,
#     description=(
#         "Useful for retrieving specific  data related to Solidity knowledge graph."
#     ),
# )

# kg_rust_solidity_tool = RetrieverTool.from_defaults(
#     retriever=kg_index_rust_retriever,
#     description=(
#         "Useful for retrieving specific  data related to Rust knowledge graph."
#     ),
# )


vector_tool = RetrieverTool.from_defaults(
    retriever=vector_index_retriever,
    description=(
        "Useful for retrieving specific context about code documentation, guides on how-to use co-pilot."
    ),
)