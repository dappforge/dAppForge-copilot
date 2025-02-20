from llama_index.core import StorageContext, load_index_from_storage, SimpleDirectoryReader, PropertyGraphIndex
from llama_index.core import PromptTemplate
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from .kg_config import fs, logger, S3_PATH_INK, S3_PATH_SUBSTRATE,S3_PATH_SOLIDITY,S3_PATH_RUST, PERSIST_DISK_PATH, text_qa_template_str
from llama_index.core.tools import RetrieverTool
from botocore.exceptions import ClientError
from llama_index.core import ServiceContext
from llama_index.graph_stores.neptune import (
    NeptuneAnalyticsPropertyGraphStore,
    NeptuneDatabasePropertyGraphStore,
)
import nest_asyncio
from llama_index.core import Settings


def load_kg_index(s3_path, fs):
    logger.info(f"Loading knowledge graph index from storage: {s3_path}")
    storage_context = StorageContext.from_defaults(persist_dir=s3_path, fs=fs)
    
    return load_index_from_storage(storage_context)


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

def load_kg_index_from_disk():
    persist_path = PERSIST_DISK_PATH
    storage_context = StorageContext.from_defaults(persist_dir=persist_path)
    return load_index_from_storage(storage_context)


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
    print("Loaded kg index from AWS Neptune")

    return index


def create_vector_embeddings():
    try:
        documents = SimpleDirectoryReader(input_files=["/home/ubuntu/dApp-merx/dApp-codegen/vector_data/dapp_content.txt"]).load_data()
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




# Load knowledge graph indices
kg_index_substrate = load_kg_index(S3_PATH_SUBSTRATE, fs)
kg_index_ink = load_kg_index(S3_PATH_INK, fs)
kg_index_neptune = load_kg_index_from_neptune()
kg_index_solidity = load_kg_index(S3_PATH_SOLIDITY, fs)
kg_index_rust = load_kg_index(S3_PATH_RUST, fs)


# Load guide document for dApp co-pilot
vector_index = create_vector_embeddings()

#vector_query_engine = vector_index.as_query_engine()

kg_index_substrate_retriever = kg_index_substrate.as_retriever()
kg_index_ink_retriever = kg_index_ink.as_retriever()
kg_index_solidity_retriever = kg_index_solidity.as_retriever()
kg_index_rust_retriever = kg_index_rust.as_retriever()


vector_index_retriever = vector_index.as_retriever()


kg_index_substrate_tool = RetrieverTool.from_defaults(
    retriever=kg_index_substrate_retriever,
    description=(
        "Useful for retrieving context data related to Substrate knowledge graph."
    ),
)

kg_index_ink_tool = RetrieverTool.from_defaults(
    retriever=kg_index_ink_retriever,
    description=(
        "Useful for retrieving specific  data related to Ink knowledge graph."
    ),
)

kg_index_solidity_tool = RetrieverTool.from_defaults(
    retriever=kg_index_solidity_retriever,
    description=(
        "Useful for retrieving specific  data related to Solidity knowledge graph."
    ),
)

kg_rust_solidity_tool = RetrieverTool.from_defaults(
    retriever=kg_index_rust_retriever,
    description=(
        "Useful for retrieving specific  data related to Rust knowledge graph."
    ),
)


vector_tool = RetrieverTool.from_defaults(
    retriever=vector_index_retriever,
    description=(
        "Useful for retrieving specific context about code documentation, guides on how-to use co-pilot."
    ),
)