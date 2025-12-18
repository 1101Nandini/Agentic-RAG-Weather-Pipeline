"""
Vector Store (Qdrant)
---------------------
This module manages the connection to the Qdrant Vector Database.
It handles:
1. Connecting to the Qdrant Cloud (or local instance).
2. Creating the collection if it doesn't exist.
3. Configuring vector parameters (Dimensions, Distance metric).
4. Providing a singleton instance of the VectorStore for the rest of the app.
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams
from langchain_qdrant import QdrantVectorStore
from app.rag.embeddings import get_embeddings

_COLLECTION_NAME = "hybrid_rag_docs"
_VECTOR_DIM = 384  
_vector_store = None


def get_vector_store():
    """
    Initializes and returns the QdrantVectorStore instance.
    Uses a singleton pattern to avoid re-connecting on every call.
    """

    global _vector_store

    # Return cached instance if available
    if _vector_store is not None:
        return _vector_store
    
    # Fetch the embedding model (needed to convert text -> vectors)
    embeddings = get_embeddings()
    
    # Initialize the low-level Qdrant Client
    # This communicates with Qdrant Cloud cluster via API key.
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    # client.delete_collection(collection_name=_COLLECTION_NAME)
    # print("collection deleted")

    # Create collection with BOTH dense & sparse vectors
    if not client.collection_exists(_COLLECTION_NAME):
        print("📁 Creating Qdrant hybrid collection...")

        client.create_collection(
            collection_name=_COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(
                    size=_VECTOR_DIM,
                    distance=Distance.COSINE,
                )
            }
            # Note: We do NOT add sparse_vectors_config here because
            # we are handling Sparse Search (BM25) locally in retriever.py
        )

    _vector_store = QdrantVectorStore(
        client=client,
        collection_name=_COLLECTION_NAME,
        embedding=embeddings,
        vector_name="dense"
    )

    return _vector_store


def get_qdrant_client():
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )


def get_collection_name():
    return _COLLECTION_NAME
