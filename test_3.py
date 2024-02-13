from pprint import pprint

import warnings

warnings.filterwarnings("ignore")


from llama_index import SimpleDirectoryReader
from llama_index.query_engine import BaseQueryEngine, RetrieverQueryEngine

from scripts.query_engine_builder import build_index


from backend import build_index_and_query_engine

documents = SimpleDirectoryReader(
    input_files=["pdfs/eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()

index = build_index(documents, rag_type="auto_merging")

chat_engine = index.as_chat_engine()

response = chat_engine.chat("what is the best way to build a career in AI?")
print(response.response)



# query_engine: BaseQueryEngine | RetrieverQueryEngine = build_index_and_query_engine(documents, rag_type="auto_merging")

# response = query_engine.query("what is the best way to build a career in AI?")

# response.print_response_stream()
