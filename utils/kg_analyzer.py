import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

def analyze_graph_store(file_path: str) -> Tuple[Set[str], Dict[str, List[Tuple[str, str]]], Dict[str, int]]:
    """
    Analyze a graph store JSON file to extract:
    - Nodes and their types
    - Edges (relationships between nodes)
    - Edge type frequencies
    """
    nodes = set()
    edges = defaultdict(list)
    edge_counts = Counter()
    
    print(f"Reading file: {file_path}")
    print(f"File size: {os.path.getsize(file_path)} bytes")
    
    try:
        with open(file_path, 'r') as f:
            print("Loading JSON data...")
            data = json.load(f)
            print(f"JSON data loaded. Top-level keys: {list(data.keys())}")
            
            if not isinstance(data, dict):
                print(f"Error: Expected dict, got {type(data)}")
                return set(), {}, {}
            
            if 'graph_dict' not in data:
                print("Error: 'graph_dict' not found in data")
                return set(), {}, {}
            
            graph = data['graph_dict']
            print(f"Number of base nodes in graph: {len(graph)}")
            
            # Process each node and its edges
            for source_node, relations in graph.items():
                nodes.add(source_node)
                
                if isinstance(relations, list):
                    for relation in relations:
                        if isinstance(relation, list) and len(relation) == 2:
                            edge_type, target_node = relation
                            edges[source_node].append((edge_type, target_node))
                            edge_counts[edge_type] += 1
                            nodes.add(target_node)  # Add target node to the set
    
    except Exception as e:
        print(f"Error analyzing graph: {str(e)}")
        return set(), {}, {}
    
    print(f"Found {len(nodes)} total nodes and {len(edge_counts)} unique edge types")
    return nodes, dict(edges), dict(edge_counts)

def sample_graph_paths(edges: Dict[str, List[Tuple[str, str]]], num_paths: int = 3, max_depth: int = 3) -> List[List[Tuple[str, str, str]]]:
    """
    Generate sample paths through the graph
    Returns list of paths, where each path is a list of (source_node, edge_type, target_node) tuples
    """
    paths = []
    if not edges:
        print("No edges found in graph")
        return paths
        
    # Get nodes that have outgoing edges
    source_nodes = [node for node, edge_list in edges.items() if edge_list]
    if not source_nodes:
        print("No nodes with outgoing edges found")
        return paths
        
    print(f"Found {len(source_nodes)} nodes with outgoing edges")
    
    attempts = 0
    max_attempts = num_paths * 3  # Allow more attempts to find valid paths
    
    while len(paths) < num_paths and attempts < max_attempts:
        attempts += 1
        current = random.choice(source_nodes)
        path = []
        visited = {current}  # Track visited nodes to avoid cycles
        
        for _ in range(max_depth):
            if current not in edges or not edges[current]:
                break
                
            # Filter out edges that would create cycles
            valid_edges = [(edge_type, target) for edge_type, target in edges[current] if target not in visited]
            if not valid_edges:
                break
                
            edge_type, target = random.choice(valid_edges)
            path.append((current, edge_type, target))
            visited.add(target)
            current = target
            
        if len(path) > 0:  # Accept paths with at least one edge
            paths.append(path)
            
    print(f"Generated {len(paths)} sample paths after {attempts} attempts")
    return paths

def analyze_kg(kg_dir: str) -> Dict:
    """
    Analyze a knowledge graph directory and return its characteristics
    """
    graph_store_path = os.path.join(kg_dir, 'graph_store.json')
    
    if not os.path.exists(graph_store_path):
        return {"error": f"Graph store not found in {kg_dir}"}
    
    try:
        nodes, edges, edge_counts = analyze_graph_store(graph_store_path)
        print("\nGenerating sample paths...")
        paths = sample_graph_paths(edges)
        
        # Group nodes by domain/category based on common words
        categorized_nodes = defaultdict(list)
        for node in nodes:
            node_lower = node.lower()
            if any(kw in node_lower for kw in ['contract', 'smart']):
                categorized_nodes['Smart Contract Nodes'].append(node)
            elif any(kw in node_lower for kw in ['storage', 'data']):
                categorized_nodes['Storage & Data Nodes'].append(node)
            elif any(kw in node_lower for kw in ['test', 'debug']):
                categorized_nodes['Testing & Debug Nodes'].append(node)
            elif any(kw in node_lower for kw in ['function', 'method']):
                categorized_nodes['Function & Method Nodes'].append(node)
            else:
                categorized_nodes['Other Nodes'].append(node)
        
        # Find most connected nodes (those with most edges)
        node_connections = {n: len(edge_list) for n, edge_list in edges.items()}
        
        return {
            "total_nodes": len(nodes),
            "total_edges": sum(edge_counts.values()),
            "node_categories": {k: len(v) for k, v in categorized_nodes.items()},
            "top_nodes": sorted(node_connections.items(), key=lambda x: x[1], reverse=True)[:10],
            "most_common_edge_types": sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            "sample_paths": paths
        }
    except Exception as e:
        return {"error": f"Error analyzing {kg_dir}: {str(e)}"}

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'local_kg_storage')
    #kg_types = ['solidity', 'rust', 'substrate', 'ink']
    kg_types = ['solidity_v3','solidity']
    
    print(f"Base directory: {base_dir}")
    print(f"Analyzing knowledge graphs: {kg_types}")
    
    for kg_type in kg_types:
        kg_dir = os.path.join(base_dir, kg_type)
        print(f"\n{'='*50}")
        print(f"Analyzing {kg_type.upper()} Knowledge Graph:")
        print(f"{'='*50}")
        
        if not os.path.exists(kg_dir):
            print(f"Error: Directory does not exist: {kg_dir}")
            continue
            
        results = analyze_kg(kg_dir)
        
        if "error" in results:
            print(results["error"])
            continue
            
        print(f"\nGraph Statistics:")
        print(f"- Total Nodes: {results['total_nodes']}")
        print(f"- Total Edges: {results['total_edges']}")
        
        print("\nNodes by Category:")
        for category, count in results['node_categories'].items():
            print(f"- {category}: {count} nodes")
        
        print("\nTop 10 Most Connected Nodes:")
        for node, count in results['top_nodes']:
            print(f"- {node} ({count} edges)")
        
        print("\nTop 10 Most Common Edge Types:")
        for edge_type, count in results['most_common_edge_types']:
            print(f"- {edge_type} (used {count} times)")
        
        print("\nSample Graph Paths:")
        for i, path in enumerate(results['sample_paths'], 1):
            print(f"\nPath {i}:")
            for source, edge_type, target in path:
                print(f"  {source} --[{edge_type}]--> {target}") 