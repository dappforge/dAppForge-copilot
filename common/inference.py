import os
import s3fs
import logging
import wandb
import networkx as nx
from jinja2 import Template
from llama_index.core import StorageContext, load_index_from_storage
from common.config import Settings
from common.utils import plot_subgraph_via_edges, load_config
from common.models import AnswerFormat
from pyvis.network import Network
from common.config import configure_settings
from llama_index.core.llms import ChatMessage
from llama_index.core import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


config = load_config()

BUCKET_NAME = config['BUCKET_NAME']
FOLDER_NAME = config['FOLDER_NAME']
S3_PATH_INK = config['S3_PATH_INK']
S3_PATH_SUBSTRATE = config['S3_PATH_SUBSTRATE']
PERSIST_DISK_PATH = config['PERSIST_DISK_PATH']

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_FILE_PATH = os.path.join(BASE_DIR, 'code_generation', 'prompts', 'code_completion.prompt')
TEXT_QA_FILE_PATH = os.path.join(BASE_DIR, 'code_generation', 'prompts', 'text_qa_template.prompt')

# Initialize S3 filesystem
fs = s3fs.S3FileSystem(anon=False)
Settings = configure_settings()



def load_template(file_path):
    """Load and return the Jinja2 template from the given file path."""
    logger.info("Loading Jinja2 template...")
    with open(file_path, 'r') as file:
        return Template(file.read())

#Load text qa template
text_qa_template = load_template(TEXT_QA_FILE_PATH)
text_qa_template_str = text_qa_template.render()

# Load the prompt template
template = load_template(PROMPT_FILE_PATH)

def load_kg_index(s3_path, fs):
    """Load the knowledge graph index from storage."""
    logger.info("Loading knowledge graph index from storage...")
    return load_index_from_storage(StorageContext.from_defaults(persist_dir=s3_path, fs=fs))

def create_query_engine(kg_index, graph_store_query_depth=1, similarity_top_k=3):
    """Create and configure the query engine."""
    logger.info("Creating and configuring the query engine...")
    text_qa_template = PromptTemplate(text_qa_template_str)

    return kg_index.as_query_engine(
        include_text=True,
        response_mode="refine",
        embedding_mode="hybrid",
        graph_store_query_depth=graph_store_query_depth,
        similarity_top_k=similarity_top_k,
        use_gpu=True,
        text_qa_template=text_qa_template
    )

def create_streaming_query_engine(kg_index, graph_store_query_depth=1, similarity_top_k=3):
    """Create and configure the query engine."""
    logger.info("Creating and configuring the query engine...")
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



# Uncomment this if you want to Load the knowledge graph index
kg_index_substrate = load_kg_index(S3_PATH_SUBSTRATE, fs)

kg_index_ink = load_kg_index(S3_PATH_INK, fs)

#kg_index = load_kg_index_from_disk()

# Create a query engine that enables streaming

def composable_graph_inference(composable_graph,prefix_code):
    """"Perform inference based on multiple knowledge graphs"""

    graph_query_engine = composable_graph.as_query_engine(
        include_text=True,
        response_mode='refine',
        graph_store_query_depth=3,
        similarity_top_k = 5,
        use_gpu=True)

    data = {
        'prefix_code': prefix_code,
    }

    query = template.render(data)
    response = graph_query_engine.query(query)
    sub_edges, subplot = plot_subgraph_via_edges(response.metadata)

    return response.response, sub_edges, subplot

def claude_inference(prefix_code, kg_name ,suffix="}"):
    """Perform inference using Claude and return the generated code, edges, and subplot."""
    logger.info("Performing inference using Claude...")
    
    # data = {'prefix_code': prefix_code}
    query = template.render({'prefix_code': prefix_code})

    # Create the query engine
    if kg_name == "substrate":
        query_engine = create_query_engine(kg_index_substrate)
    elif kg_name == "ink":
        query_engine = create_query_engine(kg_index_ink)

    response = query_engine.query(query)

    # Uncomment the line below if you want to return the subgraph
    # sub_edges, subplot = plot_subgraph_via_edges(response.metadata)
    
    # wandb.log({
    #     "Input Code": prefix_code,
    #     "Generated Code": response.response,
    #     "KG Edges": sub_edges,
    #     "Query": query
    # })

    return response.response, [], ""


def claude_inference_gradio(prefix_code, suffix="}"):
    """Perform inference using Claude and return the generated code, edges, and subplot."""
    logger.info("Performing inference using Claude...")
    
    # data = {'prefix_code': prefix_code}
    query = template.render({'prefix_code': prefix_code})
        # Create the query engine

    query_engine = create_query_engine(kg_index_ink)
    response = query_engine.query(query)

    # Uncomment the line below if you want to return the subgraph
    sub_edges, subplot = plot_subgraph_via_edges(response.metadata)
    
    # wandb.log({
    #     "Input Code": prefix_code,
    #     "Generated Code": response.response,
    #     "KG Edges": sub_edges,
    #     "Query": query
    # })

    return response.response, sub_edges, subplot

async def claude_inference_streaming(prefix_code,kg_name, suffix="}"):
    logger.info("Performing inference using Claude with streaming response...")
    query = template.render({'prefix_code': prefix_code})
    # Create the query engine
    if kg_name == "substrate":
        streaming_query_engine = create_streaming_query_engine(kg_index_substrate)
    elif kg_name == "ink":
        streaming_query_engine = create_streaming_query_engine(kg_index_ink)
        
    streaming_response = streaming_query_engine.query(query)
    for token in streaming_response.response_gen:
        print(token)
        yield token

def plot_full_kg():
    """Plot the full knowledge graph and return the HTML representation."""
    logger.info("Plotting the full knowledge graph...")
    g = kg_index_ink.get_networkx_graph()
    net = Network(
        notebook=False,
        cdn_resources="remote",
        height="500px",
        width="60%",
        select_menu=True,
        filter_menu=False,
    )
    net.from_nx(g)
    net.force_atlas_2based(central_gravity=0.015, gravity=-31)

    html = net.generate_html().replace("'", "\"")
    logger.info("HTML representation of the full knowledge graph generated.")

    return (
        '<iframe style="width: 100%; height: 600px;margin:0 auto" '
        'name="result" allow="midi; geolocation; microphone; camera; '
        'display-capture; encrypted-media;" sandbox="allow-modals allow-forms '
        'allow-scripts allow-same-origin allow-popups '
        'allow-top-navigation-by-user-activation allow-downloads" '
        'allowfullscreen="" allowpaymentrequest="" frameborder="0" '
        f'srcdoc=\'{html}\'></iframe>'
    )

# if __name__ == "__main__":
#     claude_inference("/// Macro definition")