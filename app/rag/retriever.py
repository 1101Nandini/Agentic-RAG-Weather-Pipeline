"""
Hybrid Retriever
----------------
This module implements a state-of-the-art retrieval pipeline that combines:
1. Dense Vector Search (Semantic similarity)
2. Sparse Keyword Search (BM25 for exact matches)
3. Cross-Encoder Reranking (Contextual refinement)

This approach solves the common limitations of purely vector-based RAG
by ensuring both conceptual understanding and precise keyword matching.
"""

from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from app.rag.vector_store import get_vector_store
from app.rag.loader import load_and_split_pdf


PDF_PATH = "data/Ebook-Agentic-AI.pdf"


# BM25 Singleton 
# This global variable acts as a cache.
# It ensures we only load and index the PDF once per application session.
# Without this, Streamlit would reload the PDF on every interaction (slow!).
_BM25 = None


def get_bm25_retriever(dense_k: int = 10) -> BM25Retriever:
    """
    Lazily initializes and caches the BM25 retriever.
    Prevents repeated PDF loading & indexing (Streamlit-safe).
    """
    global _BM25

    # Check if the retriever is already cached
    if _BM25 is None:
        # Load the PDF and split it into chunks
        docs = load_and_split_pdf(PDF_PATH)
        # Create the sparse index from the documents
        _BM25 = BM25Retriever.from_documents(docs)
        # Set the default number of documents to retrieve
        _BM25.k = dense_k

    return _BM25


# -----------------------------
# Hybrid Retriever
# -----------------------------
class HybridRetriever:
    """
    Hybrid Retrieval Strategy:
    1. Fetch semantic matches using Qdrant (Dense).
    2. Fetch keyword matches using BM25 (Sparse).
    3. Deduplicate results (if the same doc appears in both).
    4. Rerank the merged list using a Cross-Encoder for maximum relevance.
    """

    def __init__(self, dense_k: int = 15, final_k: int = 8):
        """
        Args:
            dense_k (int): Number of docs to fetch from EACH retriever (Vector & BM25).
            final_k (int): Number of top-tier docs to return to the LLM.
        """
        self.dense_k = dense_k
        self.final_k = final_k

        # Dense retriever (vector store)
        self.vector_store = get_vector_store()

        # Sparse retriever (cached BM25)
        self.bm25 = get_bm25_retriever(dense_k)

        # Cross-encoder reranker
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512
        )

    # -------------------------
    # Utilities
    # -------------------------
    def _deduplicate(self, docs: List[Document]) -> List[Document]:
        """
        Removes duplicate documents from the combined list.
        """
        seen = set()
        unique_docs = []

        for doc in docs:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        return unique_docs

    # -------------------------
    # Main Retrieval Logic
    # -------------------------
    def retrieve(self, query: str) -> List[Document]:
        """
        Executes hybrid retrieval + reranking.

        Flow: Query -> [Vector + BM25] -> Deduplicate -> Rerank -> Top K
        """

        # 1️ Dense semantic search
        # Finds documents with similar embeddings (meaning).
        dense_docs = self.vector_store.similarity_search(
            query, k=self.dense_k
        )

        # 2️ Sparse keyword search (LangChain 0.2+ style)
        # Finds documents with exact keyword matches.
        keyword_docs = self.bm25.invoke(query)

        # 3️ Merge & deduplicate
        docs = self._deduplicate(dense_docs + keyword_docs)
        # Limit candidate pool to 8 docs to keep reranking fast
        docs = docs[:8] 

        if not docs:
            return []

        # 4️ Cross-encoder reranking
        # Prepare pairs: (Query, Document Text) for the model to score.
        pairs = [(query, doc.page_content[:500]) for doc in docs]
        # Predict relevance scores (higher is better)
        scores = self.reranker.predict(pairs)

        # Sort documents by their new scores in descending order
        ranked_docs = [
            doc for _, doc in sorted(
                zip(scores, docs),
                key=lambda x: x[0],
                reverse=True
            )
        ]
        
        # Return only the top 'final_k' most relevant docs
        return ranked_docs[: self.final_k]


