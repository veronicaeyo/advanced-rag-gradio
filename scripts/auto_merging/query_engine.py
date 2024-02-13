from llama_index.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index import (
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
    Document,
    load_index_from_storage,
)
from llama_index.retrievers import AutoMergingRetriever
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.base import BaseIndex
from typing import List
from scripts.load_index import index_from_storage

def get_automerging_query_engine(
    index: VectorStoreIndex | BaseIndex,
    similarity_top_k=12,
    rerank_top_n=2,
) -> RetrieverQueryEngine:

    base_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank], streaming=True
    )
    return auto_merging_engine