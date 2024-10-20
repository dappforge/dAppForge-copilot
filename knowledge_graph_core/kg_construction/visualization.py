import logging
from pyvis.network import Network
from IPython.display import HTML, display

def visualize_knowledge_graph(index, output_directory):
    g = index.get_networkx_graph()
    net = Network(
        notebook=False,
        cdn_resources="remote",
        height="500px",
        width="60%",
        select_menu=True,
        filter_menu=False
    )
    net.from_nx(g)
    net.force_atlas_2based(central_gravity=0.015, gravity=-31)
    net.show(output_directory, notebook=False)
    display(HTML(filename=output_directory))
    logging.info(f"Knowledge Graph visualized and saved to {output_directory}")


def plot_full_kg(kg_index):
    """Plot the full knowledge graph and return the HTML representation."""
    g = kg_index.get_networkx_graph()
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

    return (
        '<iframe style="width: 100%; height: 600px;margin:0 auto" '
        'name="result" allow="midi; geolocation; microphone; camera; '
        'display-capture; encrypted-media;" sandbox="allow-modals allow-forms '
        'allow-scripts allow-same-origin allow-popups '
        'allow-top-navigation-by-user-activation allow-downloads" '
        'allowfullscreen="" allowpaymentrequest="" frameborder="0" '
        f'srcdoc=\'{html}\'></iframe>'
    )


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