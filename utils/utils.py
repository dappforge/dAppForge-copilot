import networkx as nx
from pyvis.network import Network
import re
import json
import os
from pathlib import Path

# Load static variables from config.json
def load_config():
    """Load configuration from config.json file."""
    base_dir = Path(__file__).parent
    config_path = base_dir / 'config.json'
    
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def extract_code_using_regex(text):
    pattern = re.compile(r'"fill_in_middle":\s*"(.*?)"\s*}', re.DOTALL)
    match = pattern.search(text)
    if match:
        # Clean up the code and handle escaped characters
        code = match.group(1)
        # Replace escaped newlines with actual newlines
        code = code.replace('\\n', '\n').replace('\\"', '"')
        return code.strip()
    return None


def extract_code_from_response(json_response):
    try:
        # Load the JSON response
        response_dict = json.loads(json_response)
        if "fill_in_middle" in response_dict:
            # Extract and clean the code
            code = response_dict["fill_in_middle"]
            code = code.replace('\\n', '\n').replace('\\"', '"').strip()
            # Replace multiple consecutive newlines with one
            cleaned_code = re.sub(r'\n\s*\n', '\n', code)
            return cleaned_code
        else:
            return None
    except json.JSONDecodeError as e:
        print("Failed to parse the response as JSON:", e)
        return None




def plot_subgraph_via_edges(metadata):
    """Plot subgraph visualization."""
    # This is a placeholder for visualization logic
    # You can implement your own visualization here if needed
    return None, None



