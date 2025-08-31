import os
import s3fs
import logging
from typing import List
from llama_index.core import StorageContext, load_index_from_storage
from utils.config import load_config  # Changed from relative to absolute import

# Load configuration
config = load_config()
BUCKET_NAME = config['BUCKET_NAME']
S3_PATH_INK = config['S3_PATH_INK']
S3_PATH_SUBSTRATE = config['S3_PATH_SUBSTRATE']
S3_PATH_SOLIDITY = config['S3_PATH_SOLIDITY']
S3_PATH_RUST = config['S3_PATH_RUST']

# Define local storage path within the project
LOCAL_KG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'local_kg_storage')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize S3 filesystem
fs = s3fs.S3FileSystem(anon=False)

def get_s3_path_for_kg(kg_name: str) -> str:
    """Get the appropriate S3 path for a given KG name."""
    kg_paths = {
        'ink': S3_PATH_INK,
        'substrate': S3_PATH_SUBSTRATE,
        'solidity': S3_PATH_SOLIDITY,
        'rust': S3_PATH_RUST
    }
    return kg_paths.get(kg_name)

def get_local_path_for_kg(kg_name: str) -> str:
    """Get the local storage path for a given KG name."""
    return os.path.join(LOCAL_KG_DIR, kg_name)

def load_kg_index_from_s3(s3_path: str, fs: s3fs.S3FileSystem):
    """Load a knowledge graph index from S3."""
    try:
        logger.info(f"Loading knowledge graph index from S3: {s3_path}")
        storage_context = StorageContext.from_defaults(persist_dir=s3_path, fs=fs)
        return load_index_from_storage(storage_context)
    except Exception as e:
        logger.error(f"Error loading knowledge graph from S3 {s3_path}: {str(e)}")
        return None

def load_kg_index_from_local(local_path: str):
    """Load a knowledge graph index from local storage."""
    try:
        logger.info(f"Loading knowledge graph index from local storage: {local_path}")
        storage_context = StorageContext.from_defaults(persist_dir=local_path)
        return load_index_from_storage(storage_context)
    except Exception as e:
        logger.error(f"Error loading knowledge graph from local storage {local_path}: {str(e)}")
        return None

def download_and_persist_kg(kg_names: List[str], force_download: bool = False):
    """Download KGs from S3 and persist them locally.
    
    Args:
        kg_names: List of KG names to download
        force_download: If True, download even if local files exist
    """
    # Create local storage directory if it doesn't exist
    os.makedirs(LOCAL_KG_DIR, exist_ok=True)
    
    for kg_name in kg_names:
        s3_path = get_s3_path_for_kg(kg_name)
        if not s3_path:
            logger.error(f"Unknown KG type: {kg_name}")
            continue

        local_path = get_local_path_for_kg(kg_name)
        
        # Skip if already downloaded and force_download is False
        if os.path.exists(local_path) and not force_download:
            logger.info(f"Knowledge graph for {kg_name} already exists locally at {local_path}. Skipping download.")
            continue

        try:
            logger.info(f"Downloading {kg_name} from S3...")
            index = load_kg_index_from_s3(s3_path, fs)
            if index:
                os.makedirs(local_path, exist_ok=True)
                index.storage_context.persist(persist_dir=local_path)
                logger.info(f"Successfully downloaded and persisted knowledge graph for {kg_name} at {local_path}")
        except Exception as e:
            logger.error(f"Error downloading/persisting knowledge graph for {kg_name}: {str(e)}")
            continue

def load_kg_indices(kg_names: List[str], use_local: bool = True) -> dict:
    """Load KG indices either from local storage or S3.
    
    Args:
        kg_names: List of KG names to load
        use_local: If True, load from local storage, otherwise load from S3
    
    Returns:
        Dictionary mapping KG names to their loaded indices
    """
    indices = {}
    
    for kg_name in kg_names:
        if use_local:
            local_path = get_local_path_for_kg(kg_name)
            if not os.path.exists(local_path):
                logger.warning(f"Local KG not found for {kg_name}. Run download_and_persist_kg first.")
                continue
            index = load_kg_index_from_local(local_path)
        else:
            s3_path = get_s3_path_for_kg(kg_name)
            if not s3_path:
                logger.error(f"Unknown KG type: {kg_name}")
                continue
            index = load_kg_index_from_s3(s3_path, fs)
        
        if index:
            indices[kg_name] = index
    
    return indices

if __name__ == "__main__":
    # List of KG types to download
    #kg_names = ['ink', 'substrate', 'solidity', 'rust']
    kg_names = ['solidity']
    # # Download all KGs from S3 and store them locally
    # # Always force re-download to ensure we have the latest version
    download_and_persist_kg(kg_names, force_download=True)
    
    # Load KGs from local storage with timing information
    import time
    start_time = time.time()
    
    indices = load_kg_indices(kg_names, use_local=True)
    
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Loaded {len(indices)} KG indices from local storage in {total_time:.2f} seconds")
