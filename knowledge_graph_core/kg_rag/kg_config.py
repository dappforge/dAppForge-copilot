import os
import s3fs
import logging
from utils.config import Settings, configure_settings
from utils.utils import load_config
from knowledge_graph_core.prompts import (
    BASE_PROMPT,
    SUBSTRATE_PROMPT,
    INK_PROMPT,
    SOLIDITY_PROMPT,
    RUST_PROMPT
)
from knowledge_graph_core.prompts.text_qa_template import TEXT_QA_TEMPLATE
from knowledge_graph_core.prompts.code_completion import CODE_COMPLETION_PROMPT
from knowledge_graph_core.prompts.stream_code_completion import STREAM_CODE_COMPLETION_PROMPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = load_config()

BUCKET_NAME = config['BUCKET_NAME']
FOLDER_NAME = config['FOLDER_NAME']
S3_PATH_INK = config['S3_PATH_INK']
S3_PATH_SUBSTRATE = config['S3_PATH_SUBSTRATE']
S3_PATH_SOLIDITY = config['S3_PATH_SOLIDITY']
S3_PATH_RUST = config['S3_PATH_RUST']

PERSIST_DISK_PATH = config['PERSIST_DISK_PATH']

fs = s3fs.S3FileSystem(anon=False)
Settings = configure_settings()

# Load prompts
text_qa_template_str = TEXT_QA_TEMPLATE
template = CODE_COMPLETION_PROMPT
stream_template = STREAM_CODE_COMPLETION_PROMPT
framework_prompts = {
    'substrate': SUBSTRATE_PROMPT,
    'ink': INK_PROMPT,
    'solidity': SOLIDITY_PROMPT,
    'rust': RUST_PROMPT,
    'base': BASE_PROMPT
}