import os
import json
import redis
import s3fs
import logging
import time
from typing import Optional, List
from llama_index.core import StorageContext, load_index_from_storage
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Redis setup
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def generate_cache_key(*args, **kwargs) -> str:
    unique_string = ''.join(args) + ''.join(f"{k}={v}" for k, v in kwargs.items())
    return unique_string

def get_cached_result(key: str) -> Optional[dict]:
    cached_result = redis_client.get(key)
    if cached_result:
        return json.loads(cached_result)
    return None

def set_cache_result(key: str, result: dict, expiry: int = 3600):
    redis_client.set(key, json.dumps(result), ex=expiry)

def invalidate_cache():
    redis_client.flushdb()  # This will clear all cache entries

# Initialize S3 filesystem
fs = s3fs.S3FileSystem(anon=False)

# Paths and settings
BUCKET_NAME = 'knowledge-graph-data'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_FILE_PATH = os.path.join(BASE_DIR, 'code_generation', 'prompts', 'code_completion.prompt')
PERSIST_DIR = '/home/ubuntu/dApp/knowledge_graph_data'

def load_kg_index(s3_path, fs):
    """Load the knowledge graph index from storage."""
    logger.info("Loading knowledge graph index from storage...")
    return load_index_from_storage(StorageContext.from_defaults(persist_dir=s3_path, fs=fs))

def persist_kg_index(index, persist_dir):
    """Persist the knowledge graph index to disk."""
    os.makedirs(persist_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=persist_dir)
    logger.info(f"Persisted knowledge graph index to {persist_dir}")

def is_persisted(persist_path):
    """Check if the knowledge graph index is already persisted to disk."""
    required_files = [
        "default__vector_store.json",
        "docstore.json",
        "graph_store.json",
        "image__vector_store.json",
        "index_store.json"
    ]
    return all(os.path.exists(os.path.join(persist_path, file)) for file in required_files)

def load_and_persist_kg(folder_names: List[str]):
    for folder_name in folder_names:
        s3_path = f"s3://{BUCKET_NAME}/{folder_name}"
        persist_path = os.path.join(PERSIST_DIR, folder_name.replace('/', '_'))

        # Check if the knowledge graph is already persisted
        if is_persisted(persist_path):
            logger.info(f"Knowledge graph for {folder_name} already persisted at {persist_path}. Skipping download.")
            continue

        # Generate a cache key
        cache_key = generate_cache_key(folder_name)

        # Check if the result is already cached
        cached_result = get_cached_result(cache_key)
        if cached_result:
            logger.info(f"Cache hit for {s3_path}. Skipping download.")
            continue

        start_time = time.time()

        # Load the knowledge graph index from S3
        index = load_kg_index(s3_path, fs)
        if index:
            # Persist the knowledge graph index to disk
            persist_kg_index(index, persist_path)

            # Cache the result
            set_cache_result(cache_key, {"persist_path": persist_path})

            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"Processed {s3_path} in {elapsed_time:.2f} seconds and cached the result.")

if __name__ == "__main__":
    # Example usage:
    folder_names = [
        "kg_gh_subset/kg_data", "ajuna-parachain", "Astar", "bifrost",
        "crust", "cumulus", "darwinia", "kg_gh_subset/kg_data", "kg",
        "Mandala-Node", "open-runtime-module-library", "peaq-network-node",
        "polkadot-sdk", "substrate_framwork/kg_data", "substrate-node-template",
        "substrate-pallets-merx", "substrate-parachain-template"
    ]
    load_and_persist_kg(folder_names)
