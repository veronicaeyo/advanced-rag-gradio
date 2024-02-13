from llama_index.node_parser import SentenceWindowNodeParser
from llama_index import (
    ServiceContext,
    VectorStoreIndex,
    Document,
    load_index_from_storage,
)
from llama_index.indices.base import BaseIndex
from typing import List
from scripts.load_index import load_index


def build_sentence_window_index(
    documents: List[Document],
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="sentence_index",
    window_size=3,
) -> VectorStoreIndex | BaseIndex:
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )

    document = Document(text="\n\n".join([doc.text for doc in documents]))

    sentence_index = load_index(document, sentence_context, save_dir)
    return sentence_index