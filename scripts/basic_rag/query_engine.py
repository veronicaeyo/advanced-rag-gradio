from llama_index.core import (
    ServiceContext,
    Document,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.indices.base import BaseIndex
from llama_index.core.query_engine import BaseQueryEngine

from typing import List

from scripts.load_index import load_index


def get_basic_rag_query_engine(
    index: VectorStoreIndex | BaseIndex,
    similarity_top_k=6,
) -> BaseQueryEngine:
    query_engine = index.as_query_engine(
        similarity_top_k=similarity_top_k, streaming=True
    )
    return query_engine
