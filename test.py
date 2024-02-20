
from pprint import pprint

import warnings
warnings.filterwarnings("ignore")


from llama_index.core import SimpleDirectoryReader, Document
from llama_index.llms.openai import OpenAI

from scripts import utils

import os
import openai


openai.api_key = utils.get_openai_api_key()

documents = SimpleDirectoryReader(
    input_files=["pdfs/eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()


# # Buuild basic rag

# basic_index = build_basic_rag(document, llm)
# query_engine = get_basic_rag_query_engine(basic_index)
# response = query_engine.query("What is the best way to build a career in AI?")
# response.print_response_stream()


# sentence_index = build_sentence_window_index(document, llm)
# query_engine = get_sentence_window_query_engine(sentence_index)
# pprint(query_engine.query("What is the best way to build a career in AI?").response)

# automerging_index = build_automerging_index(documents, llm)
# query_engine = get_automerging_query_engine(automerging_index)
# pprint(query_engine.query("What is the best way to build a career in AI?").response)




# from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from scripts.utils import get_cohere_api_key, get_openai_api_key
from scripts.basic_rag.build_index import build_basic_rag_index
from scripts.sentence_window.build_index import build_sentence_window_index
from scripts.auto_merging.build_index import build_automerging_index
from llama_index.core.node_parser import SentenceSplitter


parser = SentenceSplitter()

nodes= parser.get_nodes_from_documents(documents)

# embedding_model = CohereEmbedding(get_cohere_api_key())
embedding_model = OpenAIEmbedding(api_key=get_openai_api_key())

index = build_sentence_window_index(nodes, embed_model= embedding_model)