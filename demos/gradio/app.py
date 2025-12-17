import gradio as gr
import os
import sys
import uuid
import asyncio
import nest_asyncio
from pathlib import Path
from typing import List, Tuple

# Initialize nest_asyncio and event loop
nest_asyncio.apply()
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from knowledge_graph_core.kg_rag.inference import claude_chat_streaming
from knowledge_graph_core.kg_rag.kg_operations import kg_index_substrate, kg_index_ink, kg_index_solidity, kg_index_rust, create_query_engine
from knowledge_graph_core.kg_rag.kg_config import framework_prompts, bedrock_llm
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.llms import ChatMessage
from utils.config import configure_settings

# Configure settings
Settings = configure_settings()

# Global chat memory storage (mirrors the API implementation)
chat_memory = {}

def sync_chat_with_kg(query: str, kg_name: str, session_id: str):
    """
    Synchronous chat function that replicates claude_chat_streaming logic
    """
    try:
        # Initialize chat memory if not exists
        if session_id not in chat_memory:
            chat_store = SimpleChatStore()
            chat_memory[session_id] = ChatMemoryBuffer.from_defaults(
                chat_store=chat_store,
                token_limit=4000
            )

        # Select the appropriate index based on kg_name
        if kg_name == "substrate":
            index = kg_index_substrate
        elif kg_name == "ink":
            index = kg_index_ink
        elif kg_name == "solidity":
            index = kg_index_solidity
        elif kg_name == "rust":
            index = kg_index_rust
        else:
            raise ValueError(f"Unknown kg_name: {kg_name}")

        # Create query engine
        query_engine = create_query_engine(index)
        
        # Get context from the knowledge graph
        context_response = query_engine.query(query)
        context = context_response.response

        # Create chat messages
        messages = [
            ChatMessage(role="system", content=framework_prompts[kg_name]),
            ChatMessage(role="user", content=f"Context:\n{context}\n\nQuestion: {query}")
        ]

        # Get response using Bedrock (non-streaming for simplicity)
        response = bedrock_llm.chat(messages)
        
        return response.message.content

    except Exception as e:
        print(f"Error in chat: {str(e)}")
        return f"Error: {str(e)}"

def reset_chat_session(session_id: str) -> str:
    """Reset/clear the chat session memory"""
    if session_id in chat_memory:
        del chat_memory[session_id]
    return "Chat session reset successfully!"

def create_new_chat() -> Tuple[str, str, List[List[str]]]:
    """Create a new chat session"""
    new_session_id = str(uuid.uuid4())
    return new_session_id, "New chat session created!", []

def chat_with_kg(message: str, kg_name: str, session_id: str, chat_history: List[List[str]]) -> Tuple[str, List[List[str]]]:
    """
    Chat with the knowledge graph - synchronous version
    """
    if not message.strip():
        return "", chat_history
    
    try:
        # Generate session_id if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Add user message to chat history
        chat_history.append([message, None])
        
        # Get response using synchronous function
        response_text = sync_chat_with_kg(message, kg_name, session_id)
        
        # Update chat history with the response
        chat_history[-1][1] = response_text
        
        return "", chat_history
        
    except Exception as e:
        error_message = f"Error: {str(e)}"
        chat_history[-1][1] = error_message
        return "", chat_history

def chat_with_kg_stream(message: str, kg_name: str, session_id: str, chat_history: List[List[str]]) -> Tuple[str, List[List[str]]]:
    """
    Synchronous chat response - non-streaming version
    """
    if not message.strip():
        return "", chat_history
    
    try:
        # Generate session_id if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Add user message to chat history
        chat_history.append([message, ""])
        
        # Get response using synchronous function
        response_text = sync_chat_with_kg(message, kg_name, session_id)
        
        # Update chat history with the response
        chat_history[-1][1] = response_text
        return "", chat_history
        
    except Exception as e:
        error_message = f"Error: {str(e)}"
        chat_history[-1][1] = error_message
        return "", chat_history

# Create Gradio interface
with gr.Blocks(title="dApp Code Generation Chat", theme=gr.themes.Soft()) as iface:
    
    # Session management
    session_id = gr.State(value=str(uuid.uuid4()))
    
    gr.Markdown("# 🤖 dApp Code Generation Chat")
    gr.Markdown("Chat with AI using different blockchain knowledge graphs (Substrate, Ink!, Solidity, Rust)")
    
    with gr.Row():
        with gr.Column(scale=3):
            # Chat interface
            chatbot = gr.Chatbot(
                label="Chat History",
                height=500,
                show_label=True,
                container=True,
                bubble_full_width=False,
                type="messages"
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your message",
                    placeholder="Ask about code generation, blockchain development, or any programming questions...",
                    lines=2,
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("Clear Chat", variant="secondary")
                reset_btn = gr.Button("Reset Session", variant="secondary")
                new_chat_btn = gr.Button("New Chat", variant="secondary")
        
        with gr.Column(scale=1):
            # Knowledge Graph Selection
            kg_selector = gr.Radio(
                choices=["substrate", "ink", "solidity", "rust"],
                value="substrate",
                label="Select Knowledge Graph",
                info="Choose the blockchain framework for context"
            )
            
            # Session Info
            session_display = gr.Textbox(
                label="Session ID",
                value=str(uuid.uuid4()),
                interactive=False
            )
            
            # Status messages
            status_msg = gr.Textbox(
                label="Status",
                interactive=False,
                visible=True
            )
            
            gr.Markdown("""
            ### Available Knowledge Graphs:
            - **Substrate**: Framework for building blockchains
            - **Ink!**: Smart contract language for Substrate
            - **Solidity**: Ethereum smart contract language
            - **Rust**: General Rust programming language
            """)
    
    # Event handlers
    def handle_send(message, kg_name, session_id, chat_history):
        if not message.strip():
            return "", chat_history
        return chat_with_kg_stream(message, kg_name, session_id, chat_history)
    
    def handle_clear(chat_history):
        return []
    
    def handle_reset(session_id, chat_history):
        reset_chat_session(session_id)
        new_session_id = str(uuid.uuid4())
        return new_session_id, "Session reset! Chat memory cleared.", [], new_session_id
    
    def handle_new_chat(session_id, chat_history):
        new_session_id = str(uuid.uuid4())
        return new_session_id, "New chat session created!", [], new_session_id
    
    # Connect event handlers
    send_btn.click(
        fn=handle_send,
        inputs=[msg, kg_selector, session_id, chatbot],
        outputs=[msg, chatbot],
        api_name="send_message"
    )
    
    msg.submit(
        fn=handle_send,
        inputs=[msg, kg_selector, session_id, chatbot],
        outputs=[msg, chatbot],
        api_name="send_message_enter"
    )
    
    clear_btn.click(
        fn=handle_clear,
        inputs=[chatbot],
        outputs=[chatbot],
        api_name="clear_chat"
    )
    
    reset_btn.click(
        fn=handle_reset,
        inputs=[session_id, chatbot],
        outputs=[session_id, status_msg, chatbot, session_display],
        api_name="reset_session"
    )
    
    new_chat_btn.click(
        fn=handle_new_chat,
        inputs=[session_id, chatbot],
        outputs=[session_id, status_msg, chatbot, session_display],
        api_name="new_chat"
    )
    
    # Update session display when session_id changes
    def update_session_display(new_session_id):
        return new_session_id
    
    session_id.change(
        fn=update_session_display,
        inputs=[session_id],
        outputs=[session_display]
    )

if __name__ == "__main__":
    print("Starting Gradio app...")
    print("If you get port errors, try setting GRADIO_SERVER_PORT environment variable")
    print("Example: export GRADIO_SERVER_PORT=8080")
    
    # Try multiple launch configurations
    try:
        # First attempt: Standard launch
        iface.launch(
            server_name="0.0.0.0",
            share=False,
            debug=True,
            show_error=True,
            prevent_thread_lock=True,
            quiet=True
        )
    except Exception as e:
        print(f"First launch attempt failed: {e}")
        try:
            # Second attempt: Minimal configuration
            iface.launch(
                server_name="0.0.0.0",
                share=False,
                debug=False,
                show_error=False,
                quiet=True
            )
        except Exception as e2:
            print(f"Second launch attempt failed: {e2}")
            # Final attempt: Basic launch
            iface.launch()
