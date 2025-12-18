"""
Document Loader & Chunker 
-----------------------------------------------------
This module handles the "Extract" and "Transform" phases of the RAG pipeline:
1. Loading: Reads the raw PDF file from disk.
2. Cleaning: Removes noise (headers, excessive whitespace) to improve embedding quality.
3. Splitting: Breaks long text into smaller, overlapping chunks (tokens) for the Vector DB.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
import re


def clean_text(text: str) -> str:
    """
    Pre-processing function to clean raw text extracted from PDF.
    Removing noise here improves retrieval accuracy later.
    """
    
    # normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # remove repeating section headers"
    text = re.sub(
        r"\b\d+(\.\d+)*\s+The Core Pillars: From Perception to Execution\b",
        "",
        text,
        flags=re.IGNORECASE
    )

    return text.strip()



def load_and_split_pdf(
    pdf_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200
) -> List:
    """
    Loads a PDF file and splits it into chunks suitable for RAG.

    Args:
        pdf_path (str): Path to the PDF file
        chunk_size (int): Size of each chunk (tokens/characters)
        chunk_overlap (int): Overlap between chunks

    Returns:
        List[Document]: List of chunked LangChain Documents
    """

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Clean document text
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,          # bigger chunks
    chunk_overlap=100,        # reduce overlap
    separators=[
        "\n\n",               
        "\n",
        ". ",
    ]
)

    chunks = splitter.split_documents(documents)

    return chunks
