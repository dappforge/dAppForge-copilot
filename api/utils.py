import re
import json
import os
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

from knowledge_graph_core.kg_rag.inference import claude_inference, claude_inference_streaming
from knowledge_graph_core.kg_rag.kg_operations import (
    create_query_engine,
    create_streaming_query_engine
)
from knowledge_graph_core.kg_rag.kg_config import template, stream_template
from knowledge_graph_core.prompts import (
    SUBSTRATE_PROMPT,
    INK_PROMPT,
    SOLIDITY_PROMPT,
    RUST_PROMPT
)
from knowledge_graph_core.prompts.code_completion import CODE_COMPLETION_PROMPT
from knowledge_graph_core.prompts.stream_code_completion import STREAM_CODE_COMPLETION_PROMPT
from knowledge_graph_core.prompts.text_qa_template import TEXT_QA_TEMPLATE
from utils.config import configure_settings
from utils.utils import plot_subgraph_via_edges

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_and_trim_code_length(code: str, max_length: int = 8000) -> str:
    """Check and trim code length if necessary."""
    if len(code) > max_length:
        logger.warning(f"Code length ({len(code)}) exceeds maximum length ({max_length}). Trimming...")
        return code[:max_length]
    return code

def prepare_response(generated_code: str, sub_edges: List = None, subplot: str = None, input_tokens: int = None, output_tokens: int = None) -> Dict:
    """Prepare response dictionary."""
    return {
        "generated_code": generated_code,
        "kg_edges": sub_edges or [],
        "subplot": subplot or "",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }

def extract_value_from_generated_code(generated_code: str) -> str:
    """Extract value from generated code if it's in JSON format."""
    try:
        json_data = json.loads(generated_code)
        return json_data.get("fill_in_middle", generated_code)
    except json.JSONDecodeError:
        return generated_code

def clean_generated_code(code: str) -> str:
    """Clean and format generated code."""
    # Remove any markdown code block syntax
    code = re.sub(r'```\w*\n|```', '', code)
    # Remove leading/trailing whitespace
    code = code.strip()
    return code



