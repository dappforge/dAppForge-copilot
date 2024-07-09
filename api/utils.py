import os
import sys
import json
import bcrypt
import yaml
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import status
import logging
from llama_index.core import StorageContext, KnowledgeGraphIndex, Settings
from llama_index.readers.web import WholeSiteReader
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import StorageContext, SummaryIndex
import s3fs
from llama_index.readers.web import WholeSiteReader
from llama_index.core import SummaryIndex
from llama_index.core.indices.composability import ComposableGraph
from llama_index.core import StorageContext, load_index_from_storage
import nest_asyncio
import re
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import start_wandb_run
from common.models import CodeRequest, CodeResponse, KGCreationRequest, MergeKGRequest
from common.inference import claude_inference, composable_graph_inference, load_kg_index, plot_full_kg, claude_inference_streaming
from common.utils import extract_code_from_response, extract_code_using_regex
from caching.redis_cache import generate_cache_key, get_cached_result, set_cache_result, invalidate_cache
from code_generation.kg_construction.load_and_persist_kg import load_and_persist_kg


def load_users_from_yaml(file_path):
    """
    Loads user data from a YAML file, hashes the passwords, and creates a dictionary mapping usernames to hashed passwords.
    :param file_path: The path to the YAML file containing user data.
    :return: A dictionary where usernames are keys and hashed passwords are values.
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        users = {}
        for user in data['users']:
            hashed_password = bcrypt.hashpw(user['password'].encode('utf-8'), bcrypt.gensalt())
            users[user['username']] = hashed_password
        return users


def detect_source(url: str):
    if "github.com" in url:
        return "github"
    else:
        return "document"


def process_generated_code(generated_code: str, sub_edges: list, subplot: str) -> CodeResponse:
    """
    Process the generated code and extract the relevant information.

    Args:
        generated_code (str): The generated code to be processed.
        sub_edges (list): The list of sub-edges.
        subplot (str): The subplot.

    Returns:
        CodeResponse: The processed code response.

    """
    
    generated_code = Response(generated_code)
    print("Generated code")
    print(generated_code)
    # Extract the JSON content
    json_code = extract_code_from_response(generated_code.response)
    print("JSON code")
    print(json_code)

    if json_code:
        # Remove escape sequences for newlines and quotes to make it readable as code
        readable_code = json_code.replace('\\n', '\n').replace('\\"', '"')

        return CodeResponse(
            generated_code=str(readable_code),
            kg_edges=sub_edges,
            subgraph_plot=subplot
        )
    else:
        readable_code = extract_code_using_regex(generated_code.response)
        if readable_code:
            return CodeResponse(
                generated_code=str(readable_code),
                kg_edges=sub_edges,
                subgraph_plot=subplot
            )
        else:
            return CodeResponse(
                generated_code=str(generated_code.response),
                kg_edges=sub_edges,
                subgraph_plot=subplot
            )


def load_kg_index_from_disk(persist_path):
    storage_context = StorageContext.from_defaults(persist_dir=persist_path)

    return load_index_from_storage(storage_context)

def clean_generated_code(generated_code: str) -> str:
    cleaned_code = re.sub(r'```(json|)\s*', '', generated_code).strip()
    cleaned_code = cleaned_code.replace('\n', '').replace('\r', '').replace('\t', '')
    return cleaned_code

def extract_value_from_generated_code(generated_code: str) -> str:
    pattern = r'"fill_in_middle"\s*:\s*"((?:\\.|[^"\\])*)"'
    cleaned_text = generated_code.strip('```json\n').strip('\n```')
    match = re.search(pattern, cleaned_text)

    if match:
        value = match.group(1).replace('\\"', '"').replace('\\\\', '\\')
        return value
    return ""

def clean_and_escape_code_logic2(generated_code: str) -> str:
    # Remove code block markers
    cleaned_code = re.sub(r'```(json|)\s*', '', generated_code).strip()
    
    # Replace escaped characters, and clean newlines, carriage returns, and tabs
    escaped_code = cleaned_code.replace('\n', '').replace('\r', '').replace('\t', '')
    escaped_code = escaped_code.replace('\\', '\\\\').replace('"', '\\"')
    
    # Remove 'fill_in_middle' key-value pairs
    cleaned_code_without_fill_in_middle = re.sub(r'"fill_in_middle"\s*:\s*"[^"]*"', '', escaped_code)
    
    return cleaned_code_without_fill_in_middle

def prepare_response(generated_code: str, sub_edges: list, subplot: str) -> CodeResponse:
    """
    A function that prepares a response based on the generated code, sub edges, and subplot. 
    It tries to extract the value from the generated code and creates a result dictionary 
    based on the extracted value, sub edges, and subplot. If the value is not found, it loads 
    the generated code as JSON, gets the 'fill_in_middle', and creates the result dictionary. 
    If a JSONDecodeError occurs, it cleans and escapes the generated code logic, and returns 
    the cleaned code without 'fill_in_middle'. In case of any other exceptions, it raises an 
    HTTPException with status code 500 and the exception detail.
    """
    try:
        value = extract_value_from_generated_code(generated_code)
        if value:
            result = {
                "generated_code": value,
                "kg_edges": sub_edges,
                "subgraph_plot": subplot
            }
        else:
            json_response = json.loads(clean_generated_code(generated_code))
            fill_in_middle_code = json_response.get('fill_in_middle', '')
            result = {
                "generated_code": fill_in_middle_code,
                "kg_edges": sub_edges,
                "subgraph_plot": subplot
            }
        return CodeResponse(**result)
    except json.JSONDecodeError:
        cleaned_code_without_fill_in_middle = clean_and_escape_code_logic2(generated_code)
        # Return the remaining part as a string
        result = {
            "generated_code": cleaned_code_without_fill_in_middle.strip(", {}"),
            "kg_edges": sub_edges,
            "subgraph_plot": subplot
        }
        code_response_object = CodeResponse(**result)
        return code_response_object
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def check_and_trim_code_length(prefix_code: str, max_length: int = 340) -> str:
    # Adjust max_length if the length of prefix_code is greater than 200 characters
    if len(prefix_code) > 200:
        max_length = 230
    
    if len(prefix_code) > max_length:
        return prefix_code[-max_length:]
    
    return prefix_code

