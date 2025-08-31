from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class AnswerFormat(BaseModel):
    fill_in_middle: str

class CodeRequest(BaseModel):
    prefix_code: str
    kg_name: str  

class ChatRequest(BaseModel):
    query: str
    kg_name: str
    session_id: str

class CodeResponse(BaseModel):
    generated_code: str
    kg_edges: Optional[List] = None
    subgraph_plot: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

class ChatResponse(BaseModel):
    response: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

class KGCreationRequest(BaseModel):
    url: str
    kg_name: str

class MergeKGRequest(BaseModel):
    kg_names: List[str]

#Response class for output parsing
class Response:
    def __init__(self, response):
        self.response = response
