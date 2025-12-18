"""
Embeddings Module
-----------------
This module initializes the embedding model used for vectorization
in the RAG pipeline.

"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

_EMBEDDINGS = None  


def get_embeddings():
    """
    Initializes and returns the Hugging Face embedding model.
    Uses a singleton pattern to prevent reloading the model on every call.
    """
    global _EMBEDDINGS

    if _EMBEDDINGS is not None:
        return _EMBEDDINGS

    model_name = os.getenv(
        "EMBEDDING_MODEL",
        "BAAI/bge-small-en-v1.5"
    )

    print(f" Loading embeddings model: {model_name}")

    _EMBEDDINGS = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True}
    )

    return _EMBEDDINGS
