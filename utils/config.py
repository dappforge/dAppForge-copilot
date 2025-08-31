import os
import json
import logging
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Load static variables from config.json
def load_config():
    # Get the directory of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the full path to config.json
    config_path = os.path.join(base_dir, 'config.json')
    
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

config = load_config()

AWS_REGION = config['AWS_REGION']
LLM_MODEL = config['LLM_MODEL']
EMBED_MODEL = config['EMBED_MODEL']
print("Loaded Embed model:" + str(EMBED_MODEL))

def load_environment_variables():
    """Load environment variables from a .env file."""
    load_dotenv()
    logger.info("Environment variables loaded successfully.")

 

def configure_settings():
    """Configure the settings for LLM and embedding models."""
    Settings.llm = Bedrock(
        model=LLM_MODEL,
        region_name=AWS_REGION,
        context_size=200000,
        max_tokens=4096,
        timeout=300.0
    )
    Settings.embed_model = BedrockEmbedding(
        model_name="cohere.embed-multilingual-v3",  
        region_name=AWS_REGION,  
        client_kwargs={
            "model_id": "cohere.embed-multilingual-v3"  # Explicitly set the model_id
        }
    )

    bedrock_llm = Bedrock(
        model=LLM_MODEL,
        region_name=AWS_REGION,
        context_size=200000,
        max_tokens=4096,
        timeout=300.0
    )

    logger.info("Settings configured successfully.")
    print("Embedding model set to:" + str(Settings.embed_model))
    
    Settings.llm = bedrock_llm
    
    return Settings

def main():
    load_environment_variables()
    configure_settings()
    logger.info("Main function executed successfully.")

if __name__ == "__main__":
    main()
