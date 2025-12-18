"""
RAG Node
--------
"""

import re
from typing import Dict, List
from langchain_core.prompts import PromptTemplate
from app.llm.llm_client import get_llm
from app.rag.retriever import HybridRetriever

# -----------------------------
# 1. Cleaner
# -----------------------------
def clean_chunk(text: str) -> str:
    # Remove headers/footers
    text = re.sub(r'\d+\s+AGENTIC AI FOR EXECUTIVES', '', text, flags=re.IGNORECASE)
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if len(line) < 5 or line.isdigit():
            continue
        cleaned.append(line)
    return " ".join(cleaned)

# -----------------------------
# 2. Prompt Template
# -----------------------------
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""<|im_start|>system
You are a precise extraction assistant. 
Your task is to answer the user's question using ONLY the text provided in the "Context" section below.

RULES:
1. Do NOT use outside knowledge.
2. If the answer is not in the Context, strictly output: "The document does not provide a clear answer."
3. Answer directly and naturally. Do NOT start with "In this document", "The text states", or "According to the document".
4. If the context contains multiple relevant points, summarize them clearly.

EXAMPLE 1:
Context: "The sky is blue because of Rayleigh scattering."
Question: "Why is the sky blue?"
Answer: "The sky is blue because of Rayleigh scattering."

EXAMPLE 2:
Context: "The sky is blue."
Question: "What is the capital of France?"
Answer: "The document does not provide a clear answer."
<|im_end|>
<|im_start|>user
Context:
{context}

Question:
{question}
<|im_end|>
<|im_start|>assistant
"""
)

# -----------------------------
# LangGraph Node
# -----------------------------
def rag_node(state: Dict) -> Dict:
    query = state.get("query", "").strip()
    
    # 1. Retrieval
    retriever = HybridRetriever(dense_k=10, final_k=5)
    retrieved_docs = retriever.retrieve(query)
    
    if not retrieved_docs:
        state["answer"] = "The document does not provide a clear answer."
        state["source"] = "rag"  
        return state

    # 2. Context Building
    context_parts = []
    for doc in retrieved_docs:
        cleaned = clean_chunk(doc.page_content)
        if cleaned:
            context_parts.append(cleaned)
    
    context = "\n\n".join(context_parts)

    # 3. Generation
    llm = get_llm()
    prompt = RAG_PROMPT.format(context=context, question=query)
    
    response = llm.invoke(prompt)
    
    # 4. Post-processing
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()

    state["answer"] = response
    state["source"] = "rag"      
    state["context"] = retrieved_docs
    return state