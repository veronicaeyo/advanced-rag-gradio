from llama_index.core import (
    Document,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.indices.base import BaseIndex
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.embeddings.utils import EmbedType

from typing import List

from scripts.load_index import load_index
import os
from os import PathLike


def build_basic_rag_index(
    documents: List[Document],
    embed_model: EmbedType,
    save_dir: PathLike[str],
) -> BaseIndex:

    document = Document(text="\n\n".join([doc.text for doc in documents]))

    index = load_index(document, embed_model, save_dir)
    return index
