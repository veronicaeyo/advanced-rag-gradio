from llama_index.core import (
    VectorStoreIndex,
)
from llama_index.core.indices.base import BaseIndex

from scripts.utils import memory
from llama_index.core.chat_engine.types import ChatMode, BaseChatEngine
from llama_index.core.llms.utils import LLMType


def build_basic_rag_chat_engine(
    llm: LLMType, index: VectorStoreIndex | BaseIndex,
    similarity_top_k: int = 6,
) -> BaseChatEngine:
    chat_engine = index.as_chat_engine(
        llm=llm,
        chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT,
        verbose=True,
        similarity_top_k=similarity_top_k,
        memory=memory,
        streaming=True,
    )
    return chat_engine
