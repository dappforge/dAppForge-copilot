import streamlit as st
import os
import sys
import uuid
import time
import logging
from pathlib import Path
from typing import List, Tuple
from pyvis.network import Network
import networkx as nx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging to show in terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

# Remove local storage imports - we'll use Memgraph for everything
# from knowledge_graph_core.kg_rag.kg_operations import kg_index_substrate, kg_index_ink, kg_index_solidity, kg_index_rust, create_query_engine
from knowledge_graph_core.kg_rag.kg_config import framework_prompts, bedrock_llm
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.llms import ChatMessage
from utils.config import configure_settings
from llama_index.graph_stores.memgraph import MemgraphPropertyGraphStore
from llama_index.core import Settings
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
import re

# Memgraph connection configurations from environment variables
def get_memgraph_configs():
    """Get Memgraph configurations from environment variables"""
    return {
        "solidity": {
            "url": f"bolt://{os.getenv('MEMGRAPH_SOLIDITY_HOST')}:{os.getenv('MEMGRAPH_PORT', '7687')}",
            "username": os.getenv('MEMGRAPH_USER'),
            "password": os.getenv('MEMGRAPH_PASSWORD')
        },
        "ink": {
            "url": f"bolt://{os.getenv('MEMGRAPH_INK_HOST')}:{os.getenv('MEMGRAPH_PORT', '7687')}",
            "username": os.getenv('MEMGRAPH_USER'),
            "password": os.getenv('MEMGRAPH_PASSWORD')
        },
        "substrate": {
            "url": f"bolt://{os.getenv('MEMGRAPH_SUBSTRATE_HOST')}:{os.getenv('MEMGRAPH_PORT', '7687')}",
            "username": os.getenv('MEMGRAPH_USER'),
            "password": os.getenv('MEMGRAPH_PASSWORD')
        },
        "rust": {
            "url": f"bolt://{os.getenv('MEMGRAPH_RUST_HOST')}:{os.getenv('MEMGRAPH_PORT', '7687')}",
            "username": os.getenv('MEMGRAPH_USER'),
            "password": os.getenv('MEMGRAPH_PASSWORD')
        }
    }

# Initialize configs
MEMGRAPH_CONFIGS = get_memgraph_configs()

# Log the configuration being used (only once)
logger.info("Memgraph configurations loaded from environment variables")

def visualize_subgraph(context_data: dict):
    """Generate interactive visualization for knowledge graph context"""
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for node in context_data['nodes']:
        # Create a shortened display label
        content = node['content']
        display_label = content[:40] + "..." if len(content) > 40 else content
        
        # Add node
        G.add_node(
            node['id'],
            label=display_label
        )
    
    # Add relationships
    for rel in context_data['relationships']:
        G.add_edge(
            rel['source'],
            rel['target'],
            relationship=rel['type']
        )
    
    # Create network visualization
    net = Network(
        height="500px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True,
        cdn_resources='remote'
    )
    
    # Add nodes
    for node_id in G.nodes():
        net.add_node(
            node_id,
            label=G.nodes[node_id]['label'],
            color="#4169E1",  # Royal blue for all nodes
            size=30,
            font=dict(
                color="black",
                size=12
            )
        )
    
    # Add edges with relationship types
    for u, v, edge_data in G.edges(data=True):
        net.add_edge(
            u,
            v,
            label=edge_data.get('relationship', ''),
            color="#9575CD",
            arrows={
                'to': {
                    'enabled': True,
                    'scaleFactor': 0.5
                }
            },
            font={
                'size': 10,
                'color': '#666666',
                'align': 'horizontal'
            },
            smooth={
                'type': 'curvedCW', 
                'roundness': 0.2
            }
        )
    
    # Set physics options
    net.set_options("""
        {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -120,
                    "centralGravity": 0.015,
                    "springLength": 100,
                    "springConstant": 0.08,
                    "damping": 0.4,
                    "avoidOverlap": 1
                },
                "solver": "forceAtlas2Based",
                "stabilization": {
                    "enabled": true,
                    "iterations": 1500,
                    "updateInterval": 25
                }
            },
            "layout": {
                "improvedLayout": true
            },
            "interaction": {
                "hover": false,
                "tooltipDelay": 0,
                "hideEdgesOnDrag": true,
                "navigationButtons": true
            }
        }
    """)
    
    # Generate HTML
    html = net.generate_html()
    
    # Add styling and help text
    modified_html = html.replace(
        '</head>',
        '''
        <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/vis-network.min.js"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/vis-network.min.css" rel="stylesheet" type="text/css" />
        <style>
            #mynetwork {
                width: 100%;
                height: 500px;
                background: #ffffff;
                border: 1px solid #ddd;
                border-radius: 8px;
            }
            .graph-legend {
                padding: 10px;
                font-size: 12px;
                color: #666;
                background: #f8f9fa;
                border-radius: 4px;
                margin-bottom: 10px;
            }
            .legend-item {
                display: inline-block;
                margin-right: 15px;
            }
            .legend-color {
                display: inline-block;
                width: 12px;
                height: 12px;
                margin-right: 5px;
                border-radius: 2px;
            }
        </style>
        </head>
        '''
    ).replace(
        '<body>',
        '''<body>
        <div class="graph-legend">
            <div class="legend-item"><span class="legend-color" style="background:#4169E1"></span>Knowledge Graph Nodes</div>
            <div class="legend-item"><span class="legend-color" style="background:#9575CD"></span>Relationships</div>
        </div>
        '''
    )
    
    # Create iframe with sandbox to allow scripts but maintain security
    iframe_html = f"""
    <iframe 
        style="width: 100%; height: 550px; border: none; border-radius: 8px;" 
        sandbox="allow-scripts allow-same-origin"
        srcdoc='{modified_html.replace("'", "&apos;")}'
    ></iframe>
    """
    
    return iframe_html

# Configure settings
Settings = configure_settings()

# Initialize session state
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = {}
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "contexts" not in st.session_state:
    st.session_state.contexts = []
if "query_times" not in st.session_state:
    st.session_state.query_times = []

def sync_chat_with_kg(query: str, kg_name: str, session_id: str):
    """
    Synchronous chat function that replicates claude_chat_streaming logic
    """
    start_time = time.time()
    try:
        # Initialize chat memory if not exists (same as API implementation)
        if session_id not in st.session_state.chat_memory:
            chat_store = SimpleChatStore()
            st.session_state.chat_memory[session_id] = ChatMemoryBuffer.from_defaults(
                chat_store=chat_store,
                token_limit=4000
            )

        # Get chat memory for this session
        chat_memory = st.session_state.chat_memory[session_id]
        
        # Get Memgraph store for the selected KG
        memgraph_config = MEMGRAPH_CONFIGS.get(kg_name)
        if not memgraph_config:
            raise ValueError(f"Memgraph configuration not found for kg_name: {kg_name}")

        # Create query engine for the selected KG using Memgraph
        query_engine = get_memgraph_query_engine(kg_name, memgraph_config)
        
        # Get context from the knowledge graph
        context_response = query_engine.query(query)

        # Store source nodes and relationships for visualization
        source_nodes = []
        relationships = []
        node_id_counter = 0

        # Extract source nodes and relationships from the response
        if hasattr(context_response, 'source_nodes'):
            for node in context_response.source_nodes:
                if hasattr(node, 'node'):
                    # Get the raw text content
                    content = node.node.text
                    
                    # Check if content contains knowledge sequence data
                    if "subject -[predicate]->, object" in content:
                        # Parse knowledge sequences
                        lines = content.split('\n')
                        for line in lines:
                            if line.startswith("(") and line.endswith(")"):
                                # Remove parentheses and split by commas
                                parts = line.strip("()").split(",")
                                if len(parts) >= 2:
                                    # Create nodes for subject and object
                                    subject = parts[0].strip("'")
                                    predicate = parts[1].strip("'")
                                    object_val = parts[2].strip("'") if len(parts) > 2 else ""
                                    
                                    # Add subject node
                                    source_nodes.append({
                                        'id': node_id_counter,
                                        'content': subject,
                                        'score': 1.0
                                    })
                                    subject_id = node_id_counter
                                    node_id_counter += 1
                                    
                                    # Add object node
                                    source_nodes.append({
                                        'id': node_id_counter,
                                        'content': object_val,
                                        'score': 1.0
                                    })
                                    object_id = node_id_counter
                                    node_id_counter += 1
                                    
                                    # Add relationship
                                    relationships.append({
                                        'source': subject_id,
                                        'target': object_id,
                                        'type': predicate
                                    })
                    else:
                        # Regular node
                        source_nodes.append({
                            'id': node_id_counter,
                            'content': content,
                            'score': node.score if hasattr(node, 'score') else 1.0
                        })
                        node_id_counter += 1

        # Store visualization data
        st.session_state.contexts.append({
            'nodes': source_nodes,
            'relationships': relationships
        })

        # Create chat messages (same as API but with conversation history)
        messages = [
            ChatMessage(role="system", content=framework_prompts[kg_name])
        ]
        
        # Add conversation history from memory
        try:
            stored_messages = chat_memory.get_all()
            for msg in stored_messages:
                messages.append(ChatMessage(role=msg.role, content=msg.content))
        except Exception as e:
            print(f"Memory retrieval issue: {e}")
            pass
        
        # Add current query with context
        messages.append(ChatMessage(role="user", content=f"Context:\n{context_response.response}\n\nQuestion: {query}"))

        # Get response using Bedrock
        response = bedrock_llm.chat(messages)
        
        # Store the conversation in memory
        try:
            chat_memory.put(ChatMessage(role="user", content=query))
            chat_memory.put(ChatMessage(role="assistant", content=response.message.content))
        except Exception as e:
            print(f"Memory storage issue: {e}")
            pass
        
        end_time = time.time()
        query_time = end_time - start_time
        st.session_state.query_times.append(query_time)
        return response.message.content

    except Exception as e:
        end_time = time.time()
        query_time = end_time - start_time
        st.session_state.query_times.append(query_time)
        print(f"Error in chat: {str(e)}")
        st.session_state.contexts.append({
            'nodes': [],
            'relationships': []
        })
        return f"Error: {str(e)}"


# ===== Unified Memgraph Integration =====
class MemgraphRetriever(BaseRetriever):
    """Vector/keyword hybrid retriever over Memgraph for all knowledge graphs.

    Prefers ANN over the 'entity' vector index; falls back to name/keyword search.
    Returns TextNode snippets enriched with the matched entity name.
    """

    def __init__(self, graph_store: MemgraphPropertyGraphStore, top_k: int = 3):
        super().__init__()
        self.graph_store = graph_store
        self.top_k = top_k

    def _retrieve(self, query_bundle: QueryBundle):  # type: ignore[override]
        q_text = query_bundle.query_str if hasattr(query_bundle, "query_str") else str(query_bundle)
        q_vec = Settings.embed_model.get_text_embedding(q_text)

        # 1) Try in-DB vector search over entity embeddings
        try:
            rows = self.graph_store.structured_query(
                """
                CALL vector_search.search('entity', $k, $q)
                YIELD node AS entity, distance, similarity
                MATCH (entity)-[:MENTIONED_IN]->(doc)
                WHERE doc.text IS NOT NULL AND length(doc.text) > 100
                RETURN doc.text AS document_text, doc.id AS doc_id, entity.name AS entity_name, similarity
                ORDER BY similarity DESC
                LIMIT $k
                """,
                {"k": self.top_k * 2, "q": q_vec},
            )
            if rows:
                seen = set()
                out = []
                for r in rows:
                    doc_id = r.get("doc_id", "")
                    if doc_id in seen:
                        continue
                    seen.add(doc_id)
                    text = r.get("document_text", "")
                    ent = r.get("entity_name", "")
                    score = float(r.get("similarity", 0.0))
                    out.append(NodeWithScore(node=TextNode(text=f"[Related to: {ent}]\n\n{text}", id_=doc_id), score=score))
                    if len(out) >= self.top_k:
                        break
                if out:
                    return out
        except Exception:
            pass

        # 2) Fallback: simple name/keyword search
        keywords = [w for w in re.findall(r"[a-zA-Z0-9]+", q_text.lower()) if len(w) > 2][:3]
        if keywords:
            where = " OR ".join([f"toLower(entity.name) CONTAINS $k{i}" for i, _ in enumerate(keywords)])
            params = {f"k{i}": kw for i, kw in enumerate(keywords)}
            rows = self.graph_store.structured_query(
                f"""
                MATCH (entity:Entity)-[:MENTIONED_IN]->(doc)
                WHERE ({where}) AND doc.text IS NOT NULL AND length(doc.text) > 100
                RETURN doc.text AS document_text, doc.id AS doc_id, entity.name AS entity_name
                LIMIT $k
                """,
                {**params, "k": self.top_k},
            ) or []
            return [
                NodeWithScore(node=TextNode(text=f"[Related to: {r.get('entity_name','')}]\n\n{r.get('document_text','')}", id_=r.get("doc_id","")), score=0.8)
                for r in rows
            ]
        return []


@st.cache_resource(show_spinner=False)
def get_memgraph_query_engine(kg_name: str, config: dict) -> RetrieverQueryEngine:
    """Create a query engine for any knowledge graph using Memgraph."""
    try:
        store = MemgraphPropertyGraphStore(
            username=config["username"],
            password=config["password"], 
            url=config["url"]
        )
        retriever = MemgraphRetriever(store, top_k=3)
        synthesizer = get_response_synthesizer(response_mode="tree_summarize")
        query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=synthesizer)
        return query_engine
    except Exception as e:
        logger.error(f"Failed to create query engine for {kg_name}: {type(e).__name__}: {str(e)}")
        st.error(f"Failed to connect to {kg_name} Memgraph: {str(e)}")
        raise

def reset_chat_session():
    """Reset/clear the chat session memory"""
    if st.session_state.session_id in st.session_state.chat_memory:
        del st.session_state.chat_memory[st.session_state.session_id]
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.contexts = []  # Clear contexts
    st.session_state.query_times = []  # Clear query times
    # Clear the current KG so user can select a new one
    if "current_kg" in st.session_state:
        del st.session_state.current_kg
    st.success("Chat session reset successfully!")
    st.rerun()

def create_new_chat():
    """Create a new chat session"""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.contexts = []  # Clear contexts
    st.session_state.query_times = []  # Clear query times
    # Clear the current KG so user can select a new one
    if "current_kg" in st.session_state:
        del st.session_state.current_kg
    st.success("New chat session created!")
    st.rerun()

def main():
    st.set_page_config(
        page_title="dApp Code Generation Chat",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 dApp Code Generation Chat")
    st.markdown("Chat with AI using different blockchain knowledge graphs (Substrate, Ink!, Solidity, Rust)")
    

    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def test_memgraph_connections():
        """Test connections to all Memgraph instances and cache results"""
        connection_status = {}
        for kg_name, config in MEMGRAPH_CONFIGS.items():
            try:
                store = MemgraphPropertyGraphStore(
                    username=config["username"],
                    password=config["password"],
                    url=config["url"]
                )
                # Quick test query
                test_result = store.structured_query("MATCH (n) RETURN count(n) LIMIT 1")
                connection_status[kg_name] = f"🟢 Connected ({test_result[0]['count(n)']} nodes)"
            except Exception as e:
                connection_status[kg_name] = f"🔴 Error: {str(e)[:30]}..."
        return connection_status
    
    # Get cached connection status
    connection_status = test_memgraph_connections()
    
    for kg_name, status in connection_status.items():
        st.sidebar.text(f"{kg_name.title()}: {status}")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Settings")
        
        # Knowledge Graph Selection - only show if no messages in current session
        if not st.session_state.messages:
            kg_name = st.selectbox(
                "Select Knowledge Graph",
                ["substrate", "ink", "solidity", "rust"],
                index=0,
                help="Choose the blockchain framework for context (cannot be changed during conversation)"
            )
            # Store the selected KG in session state
            st.session_state.current_kg = kg_name
        else:
            # Show current KG as read-only when conversation has started
            kg_name = st.session_state.get("current_kg", "substrate")
            st.info(f"**Current KG:** {kg_name.title()}")
            st.caption("⚠️ Change KG requires starting a new chat")
        
        # Session Info
        st.subheader("Session Info")
        st.text(f"Session ID: {st.session_state.session_id[:8]}...")
        
        # Control buttons
        st.subheader("Controls")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reset Session", use_container_width=True):
                reset_chat_session()
        
        with col2:
            if st.button("🆕 New Chat", use_container_width=True):
                create_new_chat()
        
        # Clear chat history button
        st.subheader("Chat Controls")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.contexts = []
            st.session_state.query_times = []
            # Also clear the chat memory for this session
            if st.session_state.session_id in st.session_state.chat_memory:
                del st.session_state.chat_memory[st.session_state.session_id]
            st.rerun()
        
        # Query Performance Stats
        if st.session_state.query_times:
            st.subheader("Query Performance")
            latest_time = st.session_state.query_times[-1]
            st.metric("Latest Query", f"{latest_time:.2f}s")
            
            if len(st.session_state.query_times) > 1:
                avg_time = sum(st.session_state.query_times) / len(st.session_state.query_times)
                st.metric("Average Time", f"{avg_time:.2f}s")
                min_time = min(st.session_state.query_times)
                max_time = max(st.session_state.query_times)
                st.metric("Best Time", f"{min_time:.2f}s")
                st.metric("Slowest Time", f"{max_time:.2f}s")
    
    # Main chat area
    st.subheader(f"Chat with {kg_name.title()} Knowledge Graph")
    
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Add visualization button for assistant messages
            if message["role"] == "assistant" and i//2 < len(st.session_state.contexts):
                context = st.session_state.contexts[i//2]
                if context:  # Only show button if there's context
                    if st.button(f"🔍 Visualize Context", key=f"viz_{i}"):
                        # Display the interactive visualization
                        st.components.v1.html(
                            visualize_subgraph(context),
                            height=600,
                            scrolling=False
                        )
    
    # Chat input
    if prompt := st.chat_input("Ask about code generation, blockchain development, or any programming questions..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = sync_chat_with_kg(prompt, kg_name, st.session_state.session_id)
                st.markdown(response)
                # Add visualization button for the new response
                if st.button("🔍 Visualize Context", key=f"viz_new"):
                    if st.session_state.contexts and st.session_state.contexts[-1]:
                        # Display the interactive visualization
                        st.components.v1.html(
                            visualize_subgraph(st.session_state.contexts[-1]),
                            height=600,
                            scrolling=False
                        )
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 