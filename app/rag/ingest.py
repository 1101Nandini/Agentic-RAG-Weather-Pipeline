"""
Ingestion Pipeline 
------------------------
This module handles the "Extract, Transform, Load" workflow for RAG.
1. Extract: Load text from the source PDF.
2. Transform: Split text into chunks (handled by loader.py).
3. Load: Embed the chunks and upload them to the Qdrant Vector Database.

This script is designed to be idempotent—meaning running it multiple times
won't corrupt your database with duplicate data.
"""

from app.rag.loader import load_and_split_pdf
from app.rag.vector_store import (
    get_vector_store,
    get_qdrant_client,
    get_collection_name,
)

PDF_PATH = "data/Ebook-Agentic-AI.pdf"

def ingest_documents():
    """
    Main entry point for data ingestion.
    Checks if the vector DB is empty before ingesting to avoid duplicates.
    """

    #  Ensure collection exists FIRST
    vector_store = get_vector_store()

    client = get_qdrant_client()
    collection_name = get_collection_name()
    
    # We count how many vectors are currently in the collection.
    count = client.count(
        collection_name=collection_name,
        exact=True,
    ).count

    if count > 0:
        print("✅ Documents already embedded. Skipping ingestion.")
        return

    print("📁 Ingesting documents into Qdrant Cloud...")
    
    # Load and Split (Extract & Transform)
    documents = load_and_split_pdf(PDF_PATH)
    # Embed and Upload (Load)
    vector_store.add_documents(documents)

    print("✅ Ingestion completed.")
