"""
Test RAG Pipeline
-----------------
"""

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from app.rag.retriever import HybridRetriever
from app.rag.embeddings import get_embeddings


def test_retriever_returns_relevant_documents(monkeypatch):
    """
    Ensure retriever returns relevant documents from vector store.
    """

    embeddings = get_embeddings()

    # Create isolated in-memory Qdrant vector store
    test_store = QdrantVectorStore.from_documents(
        documents=[
            Document(page_content="LangGraph enables agentic workflows."),
            Document(page_content="Qdrant is a vector database for embeddings."),
            Document(page_content="Hybrid RAG combines retrieval and generation.")
        ],
        embedding=embeddings,
        collection_name="test_collection",
        location=":memory:"
    )

    # Monkeypatch get_vector_store to use test store
    monkeypatch.setattr(
        "app.rag.retriever.get_vector_store",
        lambda: test_store
    )

    retriever = HybridRetriever()

    results = retriever.retrieve("What is Hybrid RAG?")

    assert isinstance(results, list)
    assert len(results) > 0
    assert any(
        "Hybrid RAG" in doc.page_content
        for doc in results
    )
