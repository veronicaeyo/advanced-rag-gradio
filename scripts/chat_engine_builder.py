import warnings
from os import PathLike
from typing import List, Literal, TypedDict


from scripts.sentence_window.build_index import build_sentence_window_index
from scripts.sentence_window.chat_engine import build_sentence_window_chat_engine

from scripts.auto_merging.build_index import build_automerging_index
from scripts.auto_merging.chat_engine import build_automerging_chat_engine

from scripts.basic_rag.build_index import build_basic_rag_index
from scripts.basic_rag.chat_engine import build_basic_rag_chat_engine

from llama_index.core.llms.utils import LLMType
from llama_index.core import Document
from llama_index.core.indices.base import BaseIndex
from llama_index.core.query_engine import RetrieverQueryEngine, BaseQueryEngine
from llama_index.core.embeddings.utils import EmbedType


import nest_asyncio


nest_asyncio.apply()

warnings.filterwarnings("ignore")


class IndexParams(TypedDict):
    documents: List[Document]
    embed_model: EmbedType
    save_dir: PathLike[str]


class QueryParams(TypedDict):
    index: BaseIndex
    similarity_top_k: int
    llm: LLMType


class ChatEngineBuilder:
    def __init__(
        self,
        documents: List[Document],
        llm: LLMType,
        save_dir: PathLike[str],
        embed_model: EmbedType,
        rag_type: Literal["basic", "sentence_window", "auto_merging"] = "basic",
    ):
        self.rag_type = rag_type
        self.documents = documents
        self.save_dir = save_dir
        self.embed_model = embed_model
        self.llm = llm

    def build_index(
        self,
        window_size: int = 3,
        chunk_sizes: List[int] | None = None,
    ) -> BaseIndex:

        index_params: IndexParams = {
            "documents": self.documents,
            "embed_model": self.embed_model,
            "save_dir": self.save_dir,
        }

        index_builders = {
        "basic": lambda params: build_basic_rag_index(**params),
        "sentence_window": lambda params: build_sentence_window_index(**params, window_size=window_size),
        "auto_merging": lambda params: build_automerging_index(**params, chunk_sizes=chunk_sizes),
    }

        try:
            builder = index_builders[self.rag_type]
        except KeyError:
            raise ValueError(f"Invalid rag_type: {self.rag_type}")

        return builder(index_params)

    def build_chat_engine(
        self,
        similarity_top_k: int = 6,
        rerank_top_n: int = 2,
    ) -> BaseQueryEngine | RetrieverQueryEngine:

        query_params: QueryParams = {
            "index": self.build_index(),
            "similarity_top_k": similarity_top_k,
            "llm": self.llm
        }
        query_engines = {
            "basic": build_basic_rag_chat_engine(**query_params),
            "sentence_window": build_sentence_window_chat_engine(
                **query_params, rerank_top_n=rerank_top_n
            ),
            "auto_merging": build_automerging_chat_engine(
                **query_params, rerank_top_n=rerank_top_n
            ),
        }

        try:
            return query_engines[self.rag_type]
        except KeyError:
            raise ValueError(f"Invalid rag_type: {self.rag_type}")
