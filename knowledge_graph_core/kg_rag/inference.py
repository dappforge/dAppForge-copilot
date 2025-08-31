from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from utils.utils import plot_subgraph_via_edges
from .kg_config import template, stream_template, framework_prompts, bedrock_llm
from llama_index.core import Settings
from utils.ddb_utils import DdbTable, Session as DdbSession, Message as DdbMessage
from threading import Lock
from utils.config import configure_settings
from api.langfuse_utils import log_generation_usage
from llama_index.core.llms import ChatMessage
import os
import re

# Add Memgraph imports
from llama_index.graph_stores.memgraph import MemgraphPropertyGraphStore
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from typing import List, Dict, Any, Tuple

# Initialize chat memory storage
chat_memory = {}

# Initialize token storage (separate from chat memory)
token_storage = {}

# Memgraph connection configurations
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

# Initialize Memgraph configs
MEMGRAPH_CONFIGS = get_memgraph_configs()

# Global caches
_driver_cache = {}
_driver_lock = Lock()
memgraph_query_engines = {}

# Configure settings for embed model
Settings = configure_settings()

def extract_core_query(query: str) -> str:
    """Extract the core user query from formatted template queries"""
    # If it's a template query, try to extract the actual user intent
    if "prefix_code" in query and "Generate" in query:
        # Extract the actual user request
        lines = query.split('\n')
        for line in lines:
            if line.strip().startswith("Generate") or line.strip().startswith("Create") or line.strip().startswith("Write"):
                return line.strip()
    
    # If it's a simple query, return as is
    return query


def _get_cached_driver(bolt_url: str, username: str, password: str):
    """Get or create a cached Neo4j driver per bolt_url with thread safety."""
    key = (bolt_url, username or "", password or "")
    if key in _driver_cache:
        return _driver_cache[key]
    with _driver_lock:
        if key in _driver_cache:
            return _driver_cache[key]
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(bolt_url, auth=(username, password))
        _driver_cache[key] = driver
        return driver


def warmup_all_engines() -> None:
    """Pre-cache Memgraph drivers and query engines for all configured KGs.

    This should be called once at application startup to avoid per-request
    engine creation latency.
    """
    try:
        for _kg_name, _cfg in MEMGRAPH_CONFIGS.items():
            try:
                # Ensure driver is created and cached
                _get_cached_driver(_cfg["url"], _cfg["username"], _cfg["password"]) 
                # Ensure query engine is created and cached
                get_memgraph_query_engine(_kg_name, _cfg)
            except Exception as _e:
                print(f"Warmup skipped for {_kg_name}: {type(_e).__name__}: {str(_e)}")
    except Exception as e:
        print(f"Warmup failed: {type(e).__name__}: {str(e)}")

def memgraph_vector_retrieve(
    query: str,
    bolt_url: str,
    username: str,
    password: str,
    index_name: str = "kgtriple_embedding",
    k_triples: int = 12,
    k_chunks_per_triple: int = 2,
) -> Tuple[str, List[NodeWithScore]]:
    """
    Enhanced LlamaIndex-like retrieval over Memgraph with better relationship extraction
    and improved discovery of testing/deployment content (Memgraph compatible regex)
    EXACTLY like the working script
    """
    # 1) Embed the query using the SAME model used for ingestion
    qvec = Settings.embed_model.get_text_embedding(query)

    driver = _get_cached_driver(bolt_url, username, password)
    nodes: List[NodeWithScore] = []

    # 2) Enhanced Cypher query with Memgraph-compatible regex patterns - EXACTLY like script
    cy = """
    CALL vector_search.search($index, $k, $qvec) YIELD node, similarity
    WITH node, similarity
    // Get the actual relationship information
    OPTIONAL MATCH (node)-[:SUBJECT]->(s:Entity)
    OPTIONAL MATCH (node)-[:OBJECT]->(o:Entity)
    WITH node, similarity, s, o, node.p AS predicate
    
    // Get chunks related to the subject entity
    OPTIONAL MATCH (s)-[:MENTIONED_IN]->(d1:Doc)
    OPTIONAL MATCH (d1)-[:HAS_CHUNK]->(c1:Chunk)
    WITH node, similarity, s, o, predicate, collect(DISTINCT c1.text)[0..$k_chunks] AS s_texts
    
    // Get chunks related to the object entity  
    OPTIONAL MATCH (o)-[:MENTIONED_IN]->(d2:Doc)
    OPTIONAL MATCH (d2)-[:HAS_CHUNK]->(c2:Chunk)
    WITH node, similarity, s, o, predicate, s_texts, collect(DISTINCT c2.text)[0..$k_chunks] AS o_texts
    

    // Combine all chunk texts with quality test chunks mixed in naturally
    WITH node, similarity, s, o, predicate, 
         s_texts + o_texts AS all_texts
    
    // Get direct relationships in the knowledge graph
    OPTIONAL MATCH (s)-[rel]->(o)
    WITH node, similarity, s, o, predicate, all_texts, collect(type(rel)) AS direct_relationships
    
    RETURN 
        coalesce(s.name, s.id, "") AS subject_name,
        coalesce(s.id, "") AS subject_id,
        predicate,
        coalesce(o.name, o.id, "") AS object_name, 
        coalesce(o.id, "") AS object_id,
        all_texts AS chunk_texts,
        direct_relationships,
        similarity,
        node.id AS triple_id
    ORDER BY similarity DESC
    LIMIT $k
    """

    with driver.session() as session:
        rows = session.run(
            cy,
            index=index_name,
            k=k_triples,
            qvec=qvec,
            k_chunks=k_chunks_per_triple,
        )
        
        for r in rows:
            subject = r["subject_name"] or r["subject_id"] or ""
            predicate = r["predicate"] or ""
            obj = r["object_name"] or r["object_id"] or ""
            sim = float(r["similarity"]) if r["similarity"] is not None else 0.0
            chunks: List[str] = [t for t in (r["chunk_texts"] or []) if isinstance(t, str) and t.strip()]
            direct_rels = r["direct_relationships"] or []
            triple_id = r["triple_id"] or ""

            # Build enhanced TextNode with relationship metadata
            triple_line = f"TRIPLE: {subject} -[{predicate}]-> {obj}"
            
            # Add direct relationship information if available
            if direct_rels:
                rel_info = f"\nDIRECT_RELATIONSHIPS: {', '.join(direct_rels)}"
                triple_line += rel_info
            
            # Add chunk context with smart deduplication and filtering
            chunk_lines = ""
            if chunks:
                # Remove duplicates and filter out very short or noisy chunks
                unique_chunks = []
                seen_chunks = set()
                for c in chunks:
                    c_clean = c.strip()
                    # Skip very short chunks (< 50 chars) or duplicates
                    if len(c_clean) >= 50 and c_clean not in seen_chunks:
                        unique_chunks.append(c_clean)
                        seen_chunks.add(c_clean)
                
                # Take the best chunks up to the limit
                shown = unique_chunks[:k_chunks_per_triple]
                bullets = "\n".join([f"- {c[:300]}" for c in shown])
                if bullets:
                    chunk_lines = f"\nCONTEXT:\n{bullets}"
            
            text = f"{triple_line}{chunk_lines}".strip()

            # Create TextNode with comprehensive metadata for relationship extraction
            tnode = TextNode(
                text=text, 
                metadata={
                    "triple": (subject, predicate, obj),
                    "subject": subject,
                    "predicate": predicate, 
                    "object": obj,
                    "triple_id": triple_id,
                    "direct_relationships": direct_rels,
                    "similarity": sim,
                    "source": "memgraph_kg"
                }
            )
            nodes.append(NodeWithScore(node=tnode, score=sim))

    # Do not close cached driver

    # Build the context string for the LLM
    context_lines = []
    for nws in nodes:
        context_lines.append(nws.node.text)
    context_text = "\n\n".join(context_lines)

    return context_text, nodes


def perform_keyword_retrieval(
    query: str,
    bolt_url: str,
    username: str,
    password: str,
    k_triples: int = 6
) -> List[NodeWithScore]:
    """Additional keyword-based retrieval for richer context"""
    from neo4j import GraphDatabase
    
    # Extract keywords from query
    keywords = [w for w in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(w) > 2][:5]
    if not keywords:
        return []
    
    print(f"🔑 Keyword search using: {keywords}")
    
    # Build keyword search query
    where_conditions = " OR ".join([f"toLower(entity.name) CONTAINS '{kw}'" for kw in keywords])
    
    cy = f"""
    MATCH (entity:Entity)
    WHERE {where_conditions}
    MATCH (entity)-[:MENTIONED_IN]->(doc:Doc)
    MATCH (doc)-[:HAS_CHUNK]->(chunk:Chunk)
    WHERE chunk.text IS NOT NULL AND length(chunk.text) > 50
    RETURN 
        entity.name AS entity_name,
        entity.id AS entity_id,
        chunk.text AS chunk_text,
        doc.id AS doc_id,
        0.7 AS similarity
    ORDER BY length(chunk.text) DESC
    LIMIT $k
    """
    
    try:
        driver = GraphDatabase.driver(bolt_url, auth=(username, password))
        with driver.session() as session:
            result = session.run(cy, k=k_triples)
            rows = list(result)
            
            nodes = []
            for r in rows:
                entity_name = r["entity_name"] or r["entity_id"] or ""
                chunk_text = r["chunk_text"] or ""
                similarity = float(r["similarity"])
                
                if chunk_text.strip():
                    text = f"KEYWORD_MATCH: {entity_name}\n\n{chunk_text}"
                    tnode = TextNode(
                        text=text,
                        metadata={
                            "entity_name": entity_name,
                            "chunk_text": chunk_text,
                            "similarity": similarity,
                            "source": "keyword_search"
                        }
                    )
                    nodes.append(NodeWithScore(node=tnode, score=similarity))
        
        driver.close()
        print(f"🔑 Keyword search found {len(nodes)} additional nodes")
        return nodes
        
    except Exception as e:
        print(f"❌ Keyword search failed: {e}")
        return []


def perform_relationship_retrieval(
    query: str,
    bolt_url: str,
    username: str,
    password: str,
    k_triples: int = 4
) -> List[NodeWithScore]:
    """Additional relationship-based retrieval for richer context"""
    from neo4j import GraphDatabase
    
    # Extract potential entity names from query
    keywords = [w for w in re.findall(r"[A-Z][a-zA-Z0-9]*", query) if len(w) > 2][:3]
    if not keywords:
        return []
    
    print(f"🔗 Relationship search using: {keywords}")
    
    # Build relationship search query
    where_conditions = " OR ".join([f"toLower(s.name) CONTAINS '{kw.lower()}' OR toLower(o.name) CONTAINS '{kw.lower()}'" for kw in keywords])
    
    cy = f"""
    MATCH (s:Entity)-[r]->(o:Entity)
    WHERE {where_conditions}
    OPTIONAL MATCH (s)-[:MENTIONED_IN]->(d1:Doc)
    OPTIONAL MATCH (d1)-[:HAS_CHUNK]->(c1:Chunk)
    OPTIONAL MATCH (o)-[:MENTIONED_IN]->(d2:Doc)
    OPTIONAL MATCH (d2)-[:HAS_CHUNK]->(c2:Chunk)
    RETURN 
        s.name AS subject_name,
        o.name AS object_name,
        type(r) AS relationship_type,
        collect(DISTINCT c1.text)[0..2] AS subject_chunks,
        collect(DISTINCT c2.text)[0..2] AS object_chunks,
        0.8 AS similarity
    ORDER BY similarity DESC
    LIMIT $k
    """
    
    try:
        driver = GraphDatabase.driver(bolt_url, auth=(username, password))
        with driver.session() as session:
            result = session.run(cy, k=k_triples)
            rows = list(result)
            
            nodes = []
            for r in rows:
                subject = r["subject_name"] or ""
                obj = r["object_name"] or ""
                rel_type = r["relationship_type"] or ""
                subject_chunks = [c for c in (r["subject_chunks"] or []) if c and len(c.strip()) > 50]
                object_chunks = [c for c in (r["object_chunks"] or []) if c and len(c.strip()) > 50]
                similarity = float(r["similarity"])
                
                # Build relationship text
                rel_text = f"RELATIONSHIP: {subject} -[{rel_type}]-> {obj}"
                
                # Add context chunks
                context_lines = []
                if subject_chunks:
                    context_lines.append(f"Subject context:\n" + "\n".join([f"- {c[:300]}" for c in subject_chunks[:1]]))
                if object_chunks:
                    context_lines.append(f"Object context:\n" + "\n".join([f"- {c[:300]}" for c in object_chunks[:1]]))
                
                if context_lines:
                    rel_text += f"\n\n" + "\n\n".join(context_lines)
                
                tnode = TextNode(
                    text=rel_text,
                    metadata={
                        "subject": subject,
                        "object": obj,
                        "relationship": rel_type,
                        "similarity": similarity,
                        "source": "relationship_search"
                    }
                )
                nodes.append(NodeWithScore(node=tnode, score=similarity))
        
        driver.close()
        print(f"🔗 Relationship search found {len(nodes)} additional nodes")
        return nodes
        
    except Exception as e:
        print(f"❌ Relationship search failed: {e}")
        return []


def perform_openzeppelin_retrieval(
    query: str,
    bolt_url: str,
    username: str,
    password: str,
    k_triples: int = 3
) -> List[NodeWithScore]:
    """Specific retrieval for OpenZeppelin and important Solidity patterns"""
    from neo4j import GraphDatabase
    
    # Check if query mentions patterns that should include OpenZeppelin
    openzeppelin_triggers = [
        'contract', 'smart contract', 'token', 'erc20', 'erc721', 'erc1155', 
        'ownership', 'access control', 'pausable', 'reentrancy', 'security',
        'upgradeable', 'proxy', 'governance', 'timelock', 'multisig'
    ]
    
    query_lower = query.lower()
    should_include_openzeppelin = any(trigger in query_lower for trigger in openzeppelin_triggers)
    
    if not should_include_openzeppelin:
        print(f"🔍 No OpenZeppelin triggers found in query")
        return []
    
    print(f"🔍 OpenZeppelin retrieval triggered by: {[t for t in openzeppelin_triggers if t in query_lower]}")
    
    cy = """
    // Search for OpenZeppelin-related entities and patterns
    MATCH (entity:Entity)
    WHERE toLower(entity.name) CONTAINS 'openzeppelin' 
       OR toLower(entity.name) CONTAINS 'oz' 
       OR toLower(entity.name) CONTAINS 'erc20'
       OR toLower(entity.name) CONTAINS 'erc721'
       OR toLower(entity.name) CONTAINS 'erc1155'
       OR toLower(entity.name) CONTAINS 'ownable'
       OR toLower(entity.name) CONTAINS 'pausable'
       OR toLower(entity.name) CONTAINS 'reentrancy'
       OR toLower(entity.name) CONTAINS 'accesscontrol'
       OR toLower(entity.name) CONTAINS 'timelock'
       OR toLower(entity.name) CONTAINS 'multisig'
       OR toLower(entity.name) CONTAINS 'upgradeable'
       OR toLower(entity.name) CONTAINS 'proxy'
    MATCH (entity)-[:MENTIONED_IN]->(doc:Doc)
    MATCH (doc)-[:HAS_CHUNK]->(chunk:Chunk)
    WHERE chunk.text IS NOT NULL AND length(chunk.text) > 100
    RETURN 
        entity.name AS entity_name,
        entity.id AS entity_id,
        chunk.text AS chunk_text,
        doc.id AS doc_id,
        0.9 AS similarity
    ORDER BY similarity DESC, length(chunk.text) DESC
    LIMIT $k
    """
    
    try:
        driver = GraphDatabase.driver(bolt_url, auth=(username, password))
        with driver.session() as session:
            result = session.run(cy, k=k_triples)
            rows = list(result)
            
            nodes = []
            for r in rows:
                entity_name = r["entity_name"] or r["entity_id"] or ""
                chunk_text = r["chunk_text"] or ""
                similarity = float(r["similarity"])
                
                if chunk_text.strip():
                    text = f"OPENZEPPELIN_PATTERN: {entity_name}\n\n{chunk_text}"
                    tnode = TextNode(
                        text=text,
                        metadata={
                            "entity_name": entity_name,
                            "chunk_text": chunk_text,
                            "similarity": similarity,
                            "source": "openzeppelin_search"
                        }
                    )
                    nodes.append(NodeWithScore(node=tnode, score=similarity))
        
        driver.close()
        print(f"🔍 OpenZeppelin search found {len(nodes)} additional nodes")
        return nodes
        
    except Exception as e:
        print(f"❌ OpenZeppelin search failed: {e}")
        return []


class MemgraphRetriever(BaseRetriever):
    """Vector/keyword hybrid retriever over Memgraph for all knowledge graphs.
    Uses the proven direct Cypher approach from the working script.
    """

    def __init__(self, graph_store: MemgraphPropertyGraphStore, top_k: int = 3, bolt_url: str = None, username: str = None, password: str = None, kg_name: str = None):
        super().__init__()
        self.graph_store = graph_store
        self.top_k = top_k
        # Store the bolt URL and auth credentials for direct connection
        self.bolt_url = bolt_url or getattr(graph_store, 'url', None)
        self.username = username
        self.password = password
        self.kg_name = kg_name

    def _retrieve(self, query_bundle: QueryBundle):  # type: ignore[override]
        q_text = query_bundle.query_str if hasattr(query_bundle, "query_str") else str(query_bundle)
        print(f"🔍 MemgraphRetriever._retrieve called with query: {q_text[:100]}...")
        
        # Use the proven direct approach - EXACTLY like the working script
        if not self.bolt_url:
            raise ValueError("Bolt URL not available for Memgraph connection")
        
        print(f"🔗 Using bolt URL: {self.bolt_url}")
        
        context_text, nodes = memgraph_vector_retrieve(
            query=q_text,
            bolt_url=self.bolt_url,
            username=self.username,
            password=self.password,
            index_name="kgtriple_embedding",
            k_triples=12,  # Same as script
            k_chunks_per_triple=2  # Same as script
        )
        
        # Hybrid enrichment - add OpenZeppelin nodes (Solidity only)
        oz = []
        if (self.kg_name or "").lower().startswith("solidity"):
            print(f"🔍 Adding OpenZeppelin retrieval (solidity only)...")
            oz = perform_openzeppelin_retrieval(q_text, self.bolt_url, self.username, self.password, k_triples=3)
            nodes.extend(oz)
        
        print(f"📊 MemgraphRetriever returning {len(nodes)} nodes" + (f" (including {len(oz)} OpenZeppelin nodes)" if oz else ""))
        return nodes

def get_memgraph_query_engine(kg_name: str, config: dict) -> RetrieverQueryEngine:
    """Get or create a cached query engine per kg_name."""
    try:
        if kg_name in memgraph_query_engines:
            return memgraph_query_engines[kg_name]

        store = MemgraphPropertyGraphStore(
            username=config["username"],
            password=config["password"], 
            url=config["url"]
        )
        retriever = MemgraphRetriever(
            store, 
            top_k=3, 
            bolt_url=config["url"],
            username=config["username"],
            password=config["password"],
            kg_name=kg_name
        )
        synthesizer = get_response_synthesizer(response_mode="tree_summarize")
        query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=synthesizer)
        memgraph_query_engines[kg_name] = query_engine
        return query_engine
    except Exception as e:
        print(f"Failed to create query engine for {kg_name}: {type(e).__name__}: {str(e)}")
        raise



def composable_graph_inference(composable_graph, prefix_code):
    """
    Perform inference based on multiple knowledge graphs.
    """
    try:
        query = template.format(prefix_code=prefix_code)
        
        response = composable_graph.as_query_engine(
            include_text=True,
            response_mode='refine',
            graph_store_query_depth=3,
            similarity_top_k=5,
            use_gpu=True
        ).query(query)
        
        sub_edges, subplot = plot_subgraph_via_edges(response.metadata)
        return response.response, sub_edges, subplot
    except Exception as e:
        raise

async def claude_chat_streaming(query: str, kg_name: str, session_id: str):
    """Perform streaming chat inference using Claude."""
    try:
        print(f"🔍 Starting chat streaming for kg_name: {kg_name}, session: {session_id}")
        print(f"📝 Query: {query[:200]}...")
        
        if session_id not in chat_memory:
            print(f"🆕 Creating new chat memory for session: {session_id}")
            chat_store = SimpleChatStore()
            chat_memory[session_id] = ChatMemoryBuffer.from_defaults(
                chat_store=chat_store,
                token_limit=4000
            )

        # Get Memgraph config and create query engine on-demand (like Streamlit)
        # Config and engine are pre-warmed at startup; this is a cheap lookup
        print(f"🔗 Getting Memgraph config for {kg_name}...")
        config = MEMGRAPH_CONFIGS.get(kg_name)
        if not config:
            raise ValueError(f"Memgraph configuration not found for kg_name: {kg_name}")

        query_engine = get_memgraph_query_engine(kg_name, config)
        print(f" Executing query...")
        context_response = query_engine.query(query)
        print(f"📊 Context response received: {len(context_response.response)} characters")
        print(f"📄 Context preview: {context_response.response[:200]}...")
        
        # Debug: Show all source nodes
        if hasattr(context_response, 'source_nodes') and context_response.source_nodes:
            print(f"🔍 Found {len(context_response.source_nodes)} source nodes:")
            for i, node in enumerate(context_response.source_nodes[:3]):  # Show first 3
                print(f"   Node {i+1}: {node.node.text[:150]}...")
        else:
            print(f"🔍 No source nodes found in response")
        
        # Use the same approach as claude_inference - call LLM directly with context
        from llama_index.core.llms import ChatMessage
        
        # Rebuild raw context_text from source_nodes with relevance capping
        all_nodes = list(getattr(context_response, "source_nodes", [])) or []
        # Prefer OZ-related nodes (up to 4), then fill with others up to 10 total
        oz_nodes = [n for n in all_nodes if any(k in (n.node.text or "").lower() for k in ["openzeppelin","timelock","ownable","accesscontrol"])]
        non_oz_nodes = [n for n in all_nodes if n not in oz_nodes]
        selected = oz_nodes[:4] + non_oz_nodes[:6]
        context_text = "\n\n".join(n.node.text for n in selected if getattr(n, "node", None))
        print(f"📝 Raw context text length: {len(context_text)} characters")
        
        # Only detect/use OpenZeppelin guidance for Solidity
        is_solidity = (kg_name or "").lower().startswith("solidity")
        # Check if OpenZeppelin data was found in the context (solidity only)
        has_openzeppelin = False
        if is_solidity and hasattr(context_response, 'source_nodes'):
            has_openzeppelin = any(
                ("openzeppelin" in node.node.text.lower() or
                 "timelock" in node.node.text.lower() or
                 "ownable" in node.node.text.lower() or
                 "accesscontrol" in node.node.text.lower())
                for node in context_response.source_nodes
            )
        
        # Debug: Show what OpenZeppelin data was found
        if is_solidity and has_openzeppelin:
            openzeppelin_nodes = [node for node in context_response.source_nodes if "openzeppelin" in node.node.text.lower() or "timelock" in node.node.text.lower() or "ownable" in node.node.text.lower() or "accesscontrol" in node.node.text.lower()]
            print(f"🔍 Found {len(openzeppelin_nodes)} OpenZeppelin nodes:")
            for i, node in enumerate(openzeppelin_nodes[:3]):  # Show first 3
                print(f"   {i+1}. {node.node.text[:300]}...")
                print(f"      Contains timelock: {'timelock' in node.node.text.lower()}")
                print(f"      Contains openzeppelin: {'openzeppelin' in node.node.text.lower()}")
        
        # Also check the context response text itself
        if is_solidity and ("openzeppelin" in context_text.lower() or "timelock" in context_text.lower()):
            print(f"🔍 OpenZeppelin content found in context text")
            has_openzeppelin = True
        
        # Use context_text instead of context_response.response
        user_message = f"Context:\n{context_text}\n\nQuestion: {query}"
        if is_solidity and has_openzeppelin:
            user_message += "\n\n🚨🚨🚨 CRITICAL INSTRUCTION - YOU MUST FOLLOW THIS EXACTLY: 🚨🚨🚨\n\nOpenZeppelin patterns and implementations are available in the context above. You MUST use OpenZeppelin contracts instead of creating custom implementations.\n\nFOR TIMELOCK FUNCTIONALITY: You MUST use OpenZeppelin's TimelockController contract. Do NOT write custom timelock logic. Import and use:\nimport '@openzeppelin/contracts/governance/TimelockController.sol';\n\nYou are FORBIDDEN from creating custom timelock implementations. You MUST use OpenZeppelin's TimelockController. This is a REQUIREMENT, not a suggestion.\n\nIf you see OpenZeppelin patterns in the context, you MUST import and use them in your code. Do NOT create custom implementations."
        
        messages = [
            ChatMessage(role="system", content=framework_prompts[kg_name]),
            ChatMessage(role="user", content=user_message)
        ]
        
        print(f"🤖 Calling LLM with {len(messages)} messages...")
        print(f"📝 User message preview: {user_message[:300]}...")
        if is_solidity and has_openzeppelin:
            print(f"🚨 OpenZeppelin instruction added to user message")

        # True streaming from Bedrock (with safe fallback)
        accumulated: list[str] = []
        try:
            if hasattr(bedrock_llm, "stream_chat"):
                stream = bedrock_llm.stream_chat(messages)
            else:
                # Some versions expose a generic stream() – try it
                stream = getattr(bedrock_llm, "stream", None)
                if callable(stream):
                    stream = stream(messages)
                else:
                    stream = None

            if stream is not None:
                for event in stream:  # type: ignore[arg-type]
                    token = None
                    # LlamaIndex streaming events typically expose .delta or .text
                    if hasattr(event, "delta"):
                        token = getattr(event, "delta")
                    elif hasattr(event, "text"):
                        token = getattr(event, "text")
                    elif isinstance(event, str):
                        token = event
                    if token:
                        accumulated.append(token)
                        # yield each token immediately (true SSE streaming)
                        yield token
            else:
                # Fallback to non‑streaming call
                llm_response = bedrock_llm.chat(messages)
                response_text = llm_response.message.content
                accumulated.append(response_text)
                # Stream out the whole text once
                yield response_text
        except Exception as _stream_exc:
            # If streaming fails for any reason, attempt a non-streaming fallback
            llm_response = bedrock_llm.chat(messages)
            response_text = llm_response.message.content
            accumulated.append(response_text)
            yield response_text

        response_text = "".join(accumulated)

        # Persist this interaction to DynamoDB (best‑effort, non‑blocking for chat)
        try:
            import os
            table_name = os.getenv("DDB_CHAT_TABLE", "DAppChatSessionsTable")
            region = os.getenv("AWS_REGION", "us-east-1")
            ddb = DdbTable(table_name, region)

            # Build messages
            user_msg = DdbMessage(role="user", content=query, additional_kwargs={})
            assistant_msg = DdbMessage(role="assistant", content=response_text, additional_kwargs={})

            # Load existing or create new session
            existing = ddb.get_session(session_id)
            if existing is None:
                sess = DdbSession(id=session_id, messages=[user_msg, assistant_msg], metadata={"kg_name": kg_name})
                ddb.create_session(sess)
            else:
                msgs = list(existing.messages) if existing.messages else []
                msgs.extend([user_msg, assistant_msg])
                existing.messages = msgs
                existing.metadata = existing.metadata or {}
                existing.metadata["kg_name"] = kg_name
                ddb.update_session(existing)
        except Exception as _:
            # Do not fail the chat if persistence has issues
            pass
        
        # Calculate tokens - try to get from LLM response if available
        # Note: For streaming, actual usage might not be available until end
        input_tokens = len(query) // 4  # Heuristic for input
        output_tokens = len(response_text) // 4  # Heuristic for output
        
        # Store token counts
        token_storage[session_id] = {
            'last_input_tokens': input_tokens,
            'last_output_tokens': output_tokens
        }
        
        # Persist token counts to DynamoDB metadata (best‑effort)
        try:
            table_name = os.getenv("DDB_CHAT_TABLE", "DAppChatSessionsTable")
            region = os.getenv("AWS_REGION", "us-east-1")
            ddb = DdbTable(table_name, region)
            existing = ddb.get_session(session_id)
            if existing is None:
                # Create minimal session to store token metadata if session does not exist yet
                sess = DdbSession(
                    id=session_id,
                    messages=[],
                    metadata={
                        "kg_name": kg_name,
                        "last_input_tokens": input_tokens,
                        "last_output_tokens": output_tokens,
                        "total_input_tokens": input_tokens,
                        "total_output_tokens": output_tokens,
                        "turn_count": 1,
                        "token_source": "heuristic_char_div_4",
                    },
                )
                ddb.create_session(sess)
            else:
                meta = existing.metadata or {}
                total_in = int(meta.get("total_input_tokens", 0)) + int(input_tokens)
                total_out = int(meta.get("total_output_tokens", 0)) + int(output_tokens)
                turn_count = int(meta.get("turn_count", 0)) + 1
                meta["kg_name"] = kg_name
                meta["last_input_tokens"] = int(input_tokens)
                meta["last_output_tokens"] = int(output_tokens)
                meta["total_input_tokens"] = total_in
                meta["total_output_tokens"] = total_out
                meta["turn_count"] = turn_count
                meta["token_source"] = "heuristic_char_div_4"
                existing.metadata = meta
                ddb.update_session(existing)
        except Exception as e:
            print(f"⚠️ Failed to log usage to Langfuse: {type(e).__name__}: {str(e)}")
            print(f"⚠️ Model: {model_name}, Input tokens: {input_tokens}, Output tokens: {output_tokens}")
        
        # Already streamed above. Nothing more to yield here.

        # Best-effort: emit usage so Langfuse can price this generation
        try:
            # Use the actual Bedrock model name from config for accurate cost calculation
            from knowledge_graph_core.kg_rag.kg_config import BEDROCK_MODEL
            model_name = BEDROCK_MODEL  # "us.anthropic.claude-sonnet-4-20250514-v1:0"
            log_generation_usage(
                name='chat_code',
                model=model_name,
                input_text=query,
                output_text=response_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata={"kg_name": kg_name},
            )
        except Exception as e:
            print(f"⚠️ Failed to log usage to Langfuse: {type(e).__name__}: {str(e)}")
            print(f"⚠️ Model: {model_name}, Input tokens: {input_tokens}, Output tokens: {output_tokens}")

    except Exception as e:
        raise

def claude_inference(prefix_code: str, kg_name: str):
    """Perform inference using Claude with Memgraph."""
    try:
        print(f"🔍 Starting inference for kg_name: {kg_name}")
        query = template.format(prefix_code=prefix_code)
        print(f"📝 Generated query: {query[:200]}...")

        # Get Memgraph config and create query engine on-demand (like Streamlit)
        # Config and engine are pre-warmed at startup; this is a cheap lookup
        print(f"🔗 Getting Memgraph config for {kg_name}...")
        config = MEMGRAPH_CONFIGS.get(kg_name)
        if not config:
            raise ValueError(f"Memgraph configuration not found for kg_name: {kg_name}")

        query_engine = get_memgraph_query_engine(kg_name, config)
        print(f" Executing query...")
        context_response = query_engine.query(query)
        print(f"📊 Context response received: {len(context_response.response)} characters")
        print(f"📄 Context preview: {context_response.response[:200]}...")
        
        # Debug: Show all source nodes
        if hasattr(context_response, 'source_nodes') and context_response.source_nodes:
            print(f"🔍 Found {len(context_response.source_nodes)} source nodes:")
            for i, node in enumerate(context_response.source_nodes[:3]):  # Show first 3
                print(f"   Node {i+1}: {node.node.text[:150]}...")
        else:
            print(f"🔍 No source nodes found in response")
        
        # Use the same approach as Streamlit - call LLM directly with context
        from llama_index.core.llms import ChatMessage
        
        # Rebuild raw context_text from source_nodes with relevance capping (streaming)
        all_nodes = list(getattr(context_response, "source_nodes", [])) or []
        oz_nodes = [n for n in all_nodes if any(k in (n.node.text or "").lower() for k in ["openzeppelin","timelock","ownable","accesscontrol"])]
        non_oz_nodes = [n for n in all_nodes if n not in oz_nodes]
        selected = oz_nodes[:4] + non_oz_nodes[:6]
        context_text = "\n\n".join(n.node.text for n in selected if getattr(n, "node", None))
        print(f"📝 Raw context text length: {len(context_text)} characters")
        
        # Check if OpenZeppelin data was found in the context
        has_openzeppelin = any("openzeppelin" in node.node.text.lower() or "timelock" in node.node.text.lower() or "ownable" in node.node.text.lower() or "accesscontrol" in node.node.text.lower() for node in context_response.source_nodes) if hasattr(context_response, 'source_nodes') else False
        
        # Debug: Show what OpenZeppelin data was found
        if has_openzeppelin:
            openzeppelin_nodes = [node for node in context_response.source_nodes if "openzeppelin" in node.node.text.lower() or "timelock" in node.node.text.lower() or "ownable" in node.node.text.lower() or "accesscontrol" in node.node.text.lower()]
            print(f"🔍 Found {len(openzeppelin_nodes)} OpenZeppelin nodes:")
            for i, node in enumerate(openzeppelin_nodes[:2]):  # Show first 2
                print(f"   {i+1}. {node.node.text[:200]}...")
        
        # Also check the context response text itself
        if "openzeppelin" in context_text.lower() or "timelock" in context_text.lower():
            print(f"🔍 OpenZeppelin content found in context text")
            has_openzeppelin = True
        
        # Use context_text instead of context_response.response
        user_message = f"Context:\n{context_text}\n\nQuestion: {query}"
        if has_openzeppelin:
            user_message += "\n\nCRITICAL INSTRUCTION: OpenZeppelin patterns and implementations are available in the context above. You MUST use OpenZeppelin contracts (like TimelockController, AccessControl, Ownable, etc.) instead of creating custom implementations. Do NOT write custom time-lock logic - use OpenZeppelin's TimelockController contract."
        
        # Keep outputs concise to save tokens
        messages = [
            ChatMessage(role="system", content=framework_prompts[kg_name]),
            ChatMessage(role="user", content=user_message + "\n\nPlease output code only, no explanations.")
        ]
        
        print(f"🤖 Calling LLM with {len(messages)} messages...")
        print(f"📝 User message preview: {user_message[:300]}...")
        if has_openzeppelin:
            print(f"🚨 OpenZeppelin instruction added to user message")
        llm_response = bedrock_llm.chat(messages)
        print(f"📊 LLM response received: {len(llm_response.message.content)} characters")
        print(f"📄 LLM response preview: {llm_response.message.content[:200]}...")
        
        # DEBUG: Inspect the full response structure to find usage info
        print(f"🔍 LLM response type: {type(llm_response)}")
        print(f"🔍 LLM response attributes: {dir(llm_response)}")
        if hasattr(llm_response, 'additional_kwargs'):
            print(f"🔍 Additional kwargs: {llm_response.additional_kwargs}")
        if hasattr(llm_response, 'usage'):
            print(f"🔍 Usage info: {llm_response.usage}")
        if hasattr(llm_response, 'raw'):
            print(f"🔍 Raw response type: {type(llm_response.raw)}")
            if hasattr(llm_response.raw, 'get'):
                print(f"🔍 Raw response keys: {list(llm_response.raw.keys()) if hasattr(llm_response.raw, 'keys') else 'No keys'}")
                usage_raw = llm_response.raw.get('usage', None)
                if usage_raw:
                    print(f"🔍 Raw usage: {usage_raw}")
                    
        # Also check if there's usage in the message itself
        if hasattr(llm_response, 'message') and hasattr(llm_response.message, 'additional_kwargs'):
            print(f"🔍 Message additional kwargs: {llm_response.message.additional_kwargs}")
        
        # Try to get actual token usage from Bedrock response, fallback to heuristic
        if hasattr(llm_response, 'usage') and llm_response.usage:
            input_tokens = getattr(llm_response.usage, 'input_tokens', len(query) // 4)
            output_tokens = getattr(llm_response.usage, 'output_tokens', len(llm_response.message.content) // 4)
            print(f"🔢 Tokens (from Bedrock) - Input: {input_tokens}, Output: {output_tokens}")
        elif hasattr(llm_response, 'additional_kwargs') and llm_response.additional_kwargs:
            # Some Bedrock responses put usage in additional_kwargs
            usage = llm_response.additional_kwargs.get('usage', {})
            input_tokens = usage.get('input_tokens', len(query) // 4)
            output_tokens = usage.get('output_tokens', len(llm_response.message.content) // 4)
            print(f"🔢 Tokens (from additional_kwargs) - Input: {input_tokens}, Output: {output_tokens}")
        else:
            # Fallback to heuristic
            input_tokens = len(query) // 4
            output_tokens = len(llm_response.message.content) // 4
            print(f"🔢 Tokens (heuristic) - Input: {input_tokens}, Output: {output_tokens}")
        
        # Best-effort: emit usage so Langfuse can price this generation
        try:
            # Use the actual Bedrock model name from config for accurate cost calculation
            from knowledge_graph_core.kg_rag.kg_config import BEDROCK_MODEL
            model_name = BEDROCK_MODEL  # "us.anthropic.claude-sonnet-4-20250514-v1:0"
            log_generation_usage(
                name='generate_code',
                model=model_name,
                input_text=query,
                output_text=llm_response.message.content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata={"kg_name": kg_name},
            )
        except Exception as e:
            print(f"⚠️ Failed to log usage to Langfuse: {type(e).__name__}: {str(e)}")
            print(f"⚠️ Model: {model_name}, Input tokens: {input_tokens}, Output tokens: {output_tokens}")
            
        return llm_response.message.content, [], "", input_tokens, output_tokens

    except Exception as e:
        print(f"💥 Error in claude_inference: {type(e).__name__}: {str(e)}")
        raise



def claude_inference_gradio(prefix_code, suffix="}"):
    """Perform inference using Claude for Gradio interface."""
    try:
        query = template.format(prefix_code=prefix_code)
        # Use ink KG by default for Gradio
        query_engine = memgraph_query_engines.get("ink")
        if not query_engine:
            raise ValueError("Memgraph query engine not available for ink KG")
        response = query_engine.query(query)
        sub_edges, subplot = plot_subgraph_via_edges(response.metadata)
        return response.response, sub_edges, subplot
    except Exception as e:
        raise

async def claude_inference_streaming(prefix_code, kg_name, suffix="}"):
    
    """Perform inference using Claude with streaming response."""
    try:
        query = stream_template.format(prefix_code=prefix_code)
        
        # Get Memgraph config and create query engine on-demand (like Streamlit)
        config = MEMGRAPH_CONFIGS.get(kg_name)
        if not config:
            raise ValueError(f"Memgraph configuration not found for kg_name: {kg_name}")

        query_engine = get_memgraph_query_engine(kg_name, config)
        response = query_engine.query(query)
        # For streaming, we'll yield the response in chunks
        response_text = response.response
        chunk_size = 50  # Characters per chunk
        
        for i in range(0, len(response_text), chunk_size):
            chunk = response_text[i:i + chunk_size]
            yield chunk
            # Small delay to simulate streaming
            import asyncio
            await asyncio.sleep(0.01)

    except Exception as e:
        raise