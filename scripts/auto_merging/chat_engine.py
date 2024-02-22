from scripts.utils import memory
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.base import BaseIndex
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.llms.utils import LLMType
from llama_index.core.llms import LLM

from llama_index.core.indices.vector_store.retrievers.retriever import (
    VectorIndexRetriever,
)
from typing import cast


def build_automerging_chat_engine(
    llm: LLMType,
    index: BaseIndex,
    similarity_top_k=12,
    rerank_top_n=2,
) -> BaseChatEngine:

    base_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        cast(VectorIndexRetriever, base_retriever), index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="cross-encoder/ms-marco-TinyBERT-L-2-v2"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank], streaming=True
    )
    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        llm=cast(LLM, llm),
        memory=memory,
        query_engine=auto_merging_engine,
        verbose=True,
    )
    return chat_engine
