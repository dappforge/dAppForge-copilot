from pydantic import BaseModel
from typing import List, Optional


class AnswerFormat(BaseModel):
    fill_in_middle: str

class CodeRequest(BaseModel):
    prefix_code: str
    kg_name: str

class CodeResponse(BaseModel):
    generated_code: str
    kg_edges: list
    subgraph_plot: str

class KGCreationRequest(BaseModel):
    urls: List[str]
    kg_name: Optional[str] = None

class MergeKGRequest(BaseModel):
    kg_names: list
    prefix_code: str

#Response class for output parsing
class Response:
    def __init__(self, response):
        self.response = response