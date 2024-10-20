import networkx as nx
from pyvis.network import Network
import wandb
import re
import json
import os

# Load static variables from config.json
def load_config():
    # Get the directory of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the full path to config.json
    config_path = os.path.join(base_dir, 'config.json')
    
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




def plot_subgraph_via_edges(input_data):
    """Plot subgraph via edges from the input data and return the HTML representation."""
    edges = [value['kg_rel_texts'] for value in input_data.values() if 'kg_rel_texts' in value]
    edges = [eval(edge_str) for sublist in edges for edge_str in sublist]

    G = nx.DiGraph()
    for source, action, target in edges:
        G.add_edge(source, target, label=action)

    net = Network(
        notebook=False,
        cdn_resources="remote",
        height="500px",
        width="100%",
        select_menu=False,
        filter_menu=False,
    )
    net.from_nx(G)
    net.force_atlas_2based(central_gravity=0.015, gravity=-31)
    
    html = net.generate_html().replace("'", "\"")
    wandb.log({"Substrate KG Visualization": wandb.Html(html)})
    
    iframe_html = f"""<iframe style="width: 100%; height: 600px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera; 
    display-capture; encrypted-media;" sandbox="allow-modals allow-forms 
    allow-scripts allow-same-origin allow-popups 
    allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" 
    allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""
    
    return edges, iframe_html



