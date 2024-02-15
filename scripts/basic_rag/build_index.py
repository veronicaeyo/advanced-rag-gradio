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


def build_basic_rag_index(
    documents: List[Document],
    embed_model,
    save_dir="basic_rag_index",
) -> VectorStoreIndex | BaseIndex:

    document = Document(text="\n\n".join([doc.text for doc in documents]))

    index = load_index(document, embed_model, save_dir)
    return index
