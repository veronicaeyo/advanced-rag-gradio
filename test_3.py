
from pprint import pprint

import warnings
warnings.filterwarnings("ignore")


from llama_index import SimpleDirectoryReader, Document
from llama_index.llms import OpenAI

from scripts import utils
from scripts.basic_rag import build_basic_rag_index, get_basic_rag_query_engine
from scripts.sentence_window import build_sentence_window_index, get_sentence_window_query_engine
from scripts.auto_merging import build_automerging_index, get_automerging_query_engine

import os
import openai

from backend import build_index

openai.api_key = utils.get_openai_api_key()

documents = SimpleDirectoryReader(
    input_files=["pdfs/eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()

index = build_index(documents, rag_type="veronica")
print(index)



