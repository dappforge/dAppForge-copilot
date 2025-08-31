import os
import logging
from dotenv import load_dotenv
from llama_index.core.callbacks import CallbackManager
from langfuse.llama_index import LlamaIndexCallbackHandler

# Configure logging
logger = logging.getLogger(__name__)

load_dotenv()

def initialize_langfuse():
    """Initialize Langfuse callback handlers with error handling"""
    try:
        # Internal organization handler
        internal_handler = LlamaIndexCallbackHandler(
            public_key=os.getenv("INTERNAL_LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("INTERNAL_LANGFUSE_SECRET_KEY"),
            host=os.getenv("INTERNAL_LANGFUSE_HOST", "https://cloud.langfuse.com")
        )
        
        # Client dashboard handler
        client_handler = LlamaIndexCallbackHandler(
            public_key=os.getenv("CLIENT_LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("CLIENT_LANGFUSE_SECRET_KEY"),
            host=os.getenv("CLIENT_LANGFUSE_HOST", "https://cloud.langfuse.com")
        )
        
        # Set up callback manager with both handlers
        callback_manager = CallbackManager([internal_handler, client_handler])
        
        # Return both handlers for flushing purposes
        return {
            'internal': internal_handler,
            'client': client_handler,
            'callback_manager': callback_manager
        }
    except Exception as e:
        logger.error(f"Error initializing Langfuse: {str(e)}")
        return None

def flush_langfuse():
    """Flush any pending Langfuse traces"""
    try:
        if langfuse_handlers:
            # Flush both handlers
            langfuse_handlers['internal'].flush()
            langfuse_handlers['client'].flush()
    except Exception as e:
        logger.exception(f"Error flushing Langfuse traces: {e}")

# Create singleton instance
langfuse_handlers = initialize_langfuse()

# Export for easy access
__all__ = ['langfuse_handlers', 'flush_langfuse']

# Minimal helper to record a generation with usage so Langfuse can price it
_langfuse_client = None

def _get_langfuse_client():
    global _langfuse_client
    if _langfuse_client is None:
        try:
            # Import lazily to support SDK versions without get_client
            try:
                from langfuse import get_client, Langfuse  # type: ignore
            except Exception as e:
                logger.warning(f"Failed to import langfuse: {e}")
                return None

            # Prefer explicit constructor with provided CLIENT_/INTERNAL_ keys
            pub = (
                os.getenv("CLIENT_LANGFUSE_PUBLIC_KEY")
                or os.getenv("INTERNAL_LANGFUSE_PUBLIC_KEY")
                or os.getenv("LANGFUSE_PUBLIC_KEY")
            )
            secret = (
                os.getenv("CLIENT_LANGFUSE_SECRET_KEY")
                or os.getenv("INTERNAL_LANGFUSE_SECRET_KEY")
                or os.getenv("LANGFUSE_SECRET_KEY")
            )
            host = (
                os.getenv("CLIENT_LANGFUSE_HOST")
                or os.getenv("INTERNAL_LANGFUSE_HOST")
                or os.getenv("LANGFUSE_HOST")
                or "https://cloud.langfuse.com"
            )
            
            logger.info(f"Attempting to create Langfuse client with pub_key={'SET' if pub else 'MISSING'}, secret_key={'SET' if secret else 'MISSING'}, host={host}")
            
            if pub and secret and Langfuse is not None:
                _langfuse_client = Langfuse(public_key=pub, secret_key=secret, host=host)  # type: ignore
                logger.info("Successfully created Langfuse client with explicit credentials")
            else:
                logger.warning(f"Missing credentials - pub: {'SET' if pub else 'MISSING'}, secret: {'SET' if secret else 'MISSING'}")
                # Fallback to get_client which relies on LANGFUSE_PUBLIC_KEY/SECRET_KEY envs
                if get_client is not None:  # type: ignore
                    _langfuse_client = get_client()  # type: ignore
                    logger.info("Successfully created Langfuse client with get_client()")
                else:
                    logger.error("No Langfuse client creation method available")
                    return None
        except Exception as e:
            logger.error(f"Exception creating Langfuse client: {e}")
            return None
    return _langfuse_client

def log_generation_usage(name: str, model: str, input_text: str, output_text: str, input_tokens: int | None = None, output_tokens: int | None = None, metadata: dict | None = None) -> None:
    """Best-effort: emit a Langfuse generation with usage so costs can be computed.

    This does not change existing LlamaIndex tracing; it just ensures pricing works
    even if inference cannot derive usage from traces.
    """
    # Try method 1: Direct Langfuse client
    client = _get_langfuse_client()
    if client is None:
        logger.warning(f"Direct Langfuse client is None - trying alternative method")
        
        # Try method 2: Use the existing callback handlers that are working for traces
        if langfuse_handlers and 'internal' in langfuse_handlers:
            try:
                handler = langfuse_handlers['internal']
                logger.info(f"Trying to log via LlamaIndex callback handler: {name}, model={model}")
                
                # Try to access the underlying Langfuse client from the handler
                if hasattr(handler, 'langfuse') and handler.langfuse:
                    client = handler.langfuse
                    logger.info("Successfully got Langfuse client from callback handler")
                else:
                    logger.warning("Cannot access Langfuse client from callback handler")
                    return
            except Exception as e:
                logger.warning(f"Failed to get client from callback handler: {e}")
                return
        else:
            logger.warning(f"No langfuse_handlers available - cannot log usage for {name} with model {model}")
            return
    
    logger.info(f"Logging usage to Langfuse: {name}, model={model}, input_tokens={input_tokens}, output_tokens={output_tokens}")
    try:
        usage = {}
        if isinstance(input_tokens, int):
            usage['input'] = input_tokens
        if isinstance(output_tokens, int):
            usage['output'] = output_tokens
        client.generation(
            name=name,
            model=model,
            input=input_text,
            output=output_text,
            usage_details=usage if usage else None,
            metadata=metadata or {},
        )
        # Flush immediately to ensure the generation is sent
        client.flush()
        logger.info(f"Successfully logged usage to Langfuse for {name}")
    except Exception as e:
        # Log the error for debugging but don't fail the request
        logger.warning(f"Failed to log generation usage to Langfuse: {type(e).__name__}: {str(e)}")
        logger.warning(f"Model: {model}, Input tokens: {input_tokens}, Output tokens: {output_tokens}")