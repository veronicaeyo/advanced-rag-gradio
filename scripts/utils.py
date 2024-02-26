#!pip install python-dotenv


import os

from io import StringIO
import sys

from dotenv import load_dotenv, find_dotenv
import hashlib

from typing import List, TypedDict, Literal
from os import PathLike
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core import Document
from llama_index.core.indices.base import BaseIndex
from llama_index.core.llms.utils import LLMType
from tempfile import _TemporaryFileWrapper

import nest_asyncio

nest_asyncio.apply()

from llama_index.core.memory import ChatMemoryBuffer
memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
RAGType = Literal["basic", "sentence_window", "auto_merging"]


def get_openai_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("OPENAI_API_KEY")


def get_hf_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("HUGGINGFACE_API_KEY")


def get_cohere_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("COHERE_API_KEY")


class IndexParams(TypedDict):
    documents: List[Document]
    embed_model: EmbedType
    save_dir: PathLike[str]


class QueryParams(TypedDict):
    index: BaseIndex
    similarity_top_k: int
    llm: LLMType



def hash_file(file: _TemporaryFileWrapper) -> str:
    file_name = file.name
    unique_id = hashlib.sha256(file_name.encode()).hexdigest()
    return unique_id

class Capturing(list):
    """To capture the stdout from ReActAgent.chat with verbose=True. Taken from
    https://stackoverflow.com/questions/16571150/\
        how-to-capture-stdout-output-from-a-python-function-call
    """

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


