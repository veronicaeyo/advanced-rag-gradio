#!pip install python-dotenv


import os
from dotenv import load_dotenv, find_dotenv

from typing import List, TypedDict
from os import PathLike
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core import Document
from llama_index.core.indices.base import BaseIndex
from llama_index.core.llms.utils import LLMType


import nest_asyncio

nest_asyncio.apply()

from llama_index.core.memory import ChatMemoryBuffer
memory = ChatMemoryBuffer.from_defaults(token_limit=3900)


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
