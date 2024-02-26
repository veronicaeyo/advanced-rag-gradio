from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.indices.base import BaseIndex
from typing import List, cast
from scripts.load_index import load_index, index_from_storage
from llama_index.core.embeddings.utils import EmbedType

import os
from os import PathLike


def build_sentence_window_index(
    documents: List[Document],
    embed_model: EmbedType,
    save_dir: PathLike[str],
    window_size=3,
) -> BaseIndex:
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    # sentence_context = ServiceContext.from_defaults(
    #     llm=llm,
    #     embed_model=embed_model,
    #     node_parser=node_parser,
    # )

    document = Document(text="\n\n".join([doc.text for doc in documents]))

    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            [document], embed_model=embed_model, transformations=[node_parser]
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = cast(VectorStoreIndex, index_from_storage(save_dir))

    # sentence_index = load_index(document, sentence_context, save_dir)
    return sentence_index
