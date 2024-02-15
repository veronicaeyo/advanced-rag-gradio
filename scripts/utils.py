#!pip install python-dotenv


import os
from dotenv import load_dotenv, find_dotenv


import nest_asyncio

nest_asyncio.apply()


def get_openai_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("OPENAI_API_KEY")


def get_hf_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("HUGGINGFACE_API_KEY")



def get_cohere_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("COHERE_API_KEY")
