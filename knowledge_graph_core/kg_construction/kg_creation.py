import os
import nest_asyncio
from IPython.display import HTML, Markdown, display
from llama_index.core import (
    StorageContext, ServiceContext, KnowledgeGraphIndex, Settings
)
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.graph_stores import SimpleGraphStore
from pyvis.network import Network
from dotenv import load_dotenv
import s3fs
import re
import logging
from datetime import datetime
import time

from llama_index.core import Settings
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding
from knowledge_graph_core.prompts.kg_triplets_template import KG_TRIPLETS_TEMPLATE

def create_kg_triplet_extraction_template():
    return PromptTemplate(KG_TRIPLETS_TEMPLATE, prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT)

def create_knowledge_graph_index(documents, triplet_template, max_retries=3, initial_wait=1):
    retries = 0
    while retries < max_retries:
        try:
            logging.info("Inside create_knowledge_graph_index function")
            graph_store = SimpleGraphStore()
            storage_context = StorageContext.from_defaults(graph_store=graph_store)
            index = KnowledgeGraphIndex.from_documents(
                documents=documents,
                kg_triple_extract_template=triplet_template,
                max_triplets_per_chunk=6,
                storage_context=storage_context,
                show_progress=True,
                include_embeddings=True
            )
            logging.info("Knowledge Graph Index created")
            return index
        except Exception as e:
            if "ModelTimeoutException" in str(e):
                retries += 1
                wait_time = initial_wait * (2 ** retries)
                logging.warning(f"ModelTimeoutException occurred. Retrying {retries}/{max_retries} in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.error("An error occurred while creating the Knowledge Graph Index: %s", str(e))
                return None
    logging.error("Failed to create Knowledge Graph Index after multiple retries")
    return None