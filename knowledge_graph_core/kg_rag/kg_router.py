from llama_index.core import Settings
from llama_index.llms.anthropic import Anthropic
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import RetrieverTool, ToolMetadata
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.chat_engine import ContextChatEngine
from typing import AsyncGenerator
import asyncio

class LLMQueryRouter:
    def __init__(self, kg_index_substrate, kg_index_ink, kg_index_solidity, kg_index_rust, vector_index):
        # Initialize Claude 3.5 for code explanations
        self.code_explanation_llm = Anthropic(
            model="claude-3-5-sonnet-20241022-v2:0",
            max_tokens=4096,
        )
        
        # Store indices
        self.kg_index_substrate = kg_index_substrate
        self.kg_index_ink = kg_index_ink
        self.kg_index_solidity = kg_index_solidity
        self.kg_index_rust = kg_index_rust
        self.vector_index = vector_index
    
    def get_router(self, kg_name: str):
        """Creates a router with better tool descriptions for automatic selection"""
        if kg_name == "substrate":
            kg_retriever = self.kg_index_substrate.as_retriever(similarity_top_k=5)
            kg_description = "Use this for generating Substrate code, handling blockchain operations, and technical implementation. Best for code generation and blockchain tasks."
        elif kg_name == "ink":
            kg_retriever = self.kg_index_ink.as_retriever(similarity_top_k=5)
            kg_description = "Use this for generating Ink smart contracts, handling contract operations, and technical implementation. Best for code generation and smart contract tasks."
        elif kg_name == "solidity":
            kg_retriever = self.kg_index_solidity.as_retriever(similarity_top_k=5)
            kg_description = "Use this for generating Solidity smart contracts, handling contract operations, and technical implementation. Best for code generation and smart contract tasks."
        elif kg_name == "rust":
            kg_retriever = self.kg_index_rust.as_retriever(similarity_top_k=5)
            kg_description = "Use this for generating Rust smart contracts, handling contract operations, and technical implementation. Best for code generation and smart contract tasks."
        else:
            raise ValueError(f"Unknown kg_name: {kg_name}")

        code_explanation_retriever = self.vector_index.as_retriever(similarity_top_k=5)
        
        return RouterRetriever(
            selector=LLMSingleSelector.from_defaults(),
            retriever_tools=[
                RetrieverTool(
                    retriever=kg_retriever,
                    metadata=ToolMetadata(
                        name="code_generation",
                        description=kg_description
                    )
                ),
                RetrieverTool(
                    retriever=code_explanation_retriever,
                    metadata=ToolMetadata(
                        name="code_explanation",
                        description="Use this for explaining code, understanding documentation, clarifying technical concepts, and helping understand how code works. Best for explanations and understanding."
                    )
                )
            ]
        )

    async def get_chat_engine(self, query: str, kg_name: str, memory=None):
        """Creates chat engine that automatically selects the right tool and LLM"""
        router = self.get_router(kg_name)
        
        # Get retrieved nodes
        retrieved_nodes = router.retrieve(query)
        
        # Check if any of the retrieved tools are for code explanation
        is_code_explanation = any(
            hasattr(node, 'metadata') and 
            isinstance(node.metadata, dict) and 
            node.metadata.get("name") == "code_explanation" 
            for node in retrieved_nodes
        )
        
        # Select LLM based on the retrieval
        llm = self.code_explanation_llm if is_code_explanation else Settings.llm
            
        return ContextChatEngine.from_defaults(
            retriever=router,
            memory=memory,
            llm=llm,
            streaming=True  # Ensure streaming is enabled
        )

def router_chat_streaming(retriever: RouterRetriever, memory) -> ContextChatEngine:
    """Creates a streaming chat engine with the given retriever and memory"""
    return ContextChatEngine.from_defaults(
        retriever=retriever,
        memory=memory,
        streaming=True  # Ensure streaming is enabled
    )