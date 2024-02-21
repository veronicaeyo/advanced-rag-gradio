from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.base import BaseIndex
from llama_index.core.query_engine import BaseQueryEngine


def build_sentence_window_chat_engine(
    index: VectorStoreIndex | BaseIndex,
    similarity_top_k=6,
    rerank_top_n=2,
) -> BaseQueryEngine:
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="cross-encoder/ms-marco-TinyBERT-L-2-v2"
    )

    sentence_window_engine = index.as_query_engine(
        similarity_top_k=similarity_top_k,
        node_postprocessors=[postproc, rerank],
        streaming=True,
    )
    return sentence_window_engine
