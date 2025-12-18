#  Agentic RAG & Weather Pipeline

An intelligent AI agent built with **LangGraph** that autonomously routes user queries to the best available tool: a **Real-Time Weather API** or a **Hybrid RAG System** for document analysis.

---

##  Features

* **Intelligent Routing:** LLM-based semantic router accurately distinguishes between weather inquiries and knowledge-base questions based on user intent.
* **Hybrid RAG Architecture:** Combines **Dense Vector Search** (Semantic) with **BM25** (Keyword) and **Cross-Encoder Reranking** to ensure high-precision retrieval.
* **Real-Time Weather:** Fetches live data using OpenWeatherMap API with structured output.
* **CPU Optimized:** Designed to run efficiently on standard hardware using a lightweight 3B instruct model (`Qwen/Qwen2.5-3B`) with constrained generation.
* **Robust Evaluation:** Integrated **LangSmith** tracing to debug and evaluate agent performance.
---

##  Architecture

The pipeline follows a graph-based workflow managed by LangGraph:

graph LR

    A [User Query] --> B{Decision Node}
    
    B -- "Weather Keywords" --> C[Weather Node]

    B -- "General/Document" --> D[RAG Node]
    
    C --> E[Final Answer]
    
    D --> E


1. Decision Node: Analyzes query intent using regex patterns (Speed & Accuracy > Latency).
2. Weather Node: Calls OpenWeatherMap and returns a structured Document.
3. RAG Node:
   - Retrieves top-k chunks via Qdrant (Dense) + BM25 (Sparse).
   - Reranks results using a Cross-Encoder.
   - Generates a strict, grounded response using the LLM.


## Tech Stack
- Orchestration: LangGraph, LangChain
- LLM: Hugging Face (Qwen/Qwen2.5-3B-Instruct)
- Vector DB: Qdrant (Hybrid Search)
- Embeddings: BAAI/bge-small-en-v1.5
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
- API: OpenWeatherMap
- Observability: LangSmith
- UI: Streamlit

## Setup & Installation

### 1 Clone the Repository

- git clone <your-repo-url>
- cd ai-agentic-pipeline


### 2 Environment Configuration

**Create a .env file in the root directory:**

**LLM & Hugging Face**

- HF_TOKEN=your_huggingface_token

- HF_MODEL_NAME=Qwen/Qwen2.5-3B-Instruct

**Vector DB (Qdrant)**

- QDRANT_URL=your_qdrant_url

- QDRANT_API_KEY=your_qdrant_key

**Weather API**

- OPENWEATHER_API_KEY=your_openweather_key

**LangSmith Tracing**

- LANGCHAIN_TRACING_V2=true

- LANGCHAIN_ENDPOINT="[https://api.smith.langchain.com](https://api.smith.langchain.com)"

- LANGCHAIN_API_KEY=your_langsmith_key

- LANGCHAIN_PROJECT="agentic-rag-pipeline"



### 3 Installation (Local)

It is recommended to use a virtual environment.

- python -m venv venv

- source venv/bin/activate  # On Windows: venv\Scripts\activate

- pip install -r requirements.txt

### How to Run
**Option A: Using Docker**

This ensures the app runs in an isolated environment with all dependencies fixed.

1. Build the Image:
docker build -t agentic-rag-app .

2. Run the Container:
docker run -p 8501:8501 --env-file .env agentic-rag-app

3. Access: Open http://localhost:8501 in your browser.


**Option B: Running Locally**

streamlit run streamlit_app.py

###  Data Source

The RAG system is currently indexed on **`data/Ebook-Agentic-AI.pdf`**.
This document covers:
* Introduction to Agentic AI
* Anatomy of an Agentic AI System
* Multi-Agent Systems
* Orchestrating Agentic AI Systems
* Practical Applications of Agentic AI

**Note:** You can replace this file with any other PDF by placing it in the `data/` directory and restarting the application.


### Testing
The project includes comprehensive unit tests using pytest and unittest.mock.

Run the test suite:
**pytest**

### What is tested?

- Graph Routing: Ensures queries route to the correct node.
- RAG Logic: Verifies retrieval returns relevant documents.
- Weather API: Mocks API responses to test success/failure handling without using credits.

## Design Decisions & Trade-offs

### 1. LLM-Based Decision Routing
I implemented a **Semantic Router** using a lightweight LLM call (`Qwen/Qwen2.5-3B`) instead of a simple Keyword/Regex router.

* **Reason:** While regex is faster, it is brittle. A simple keyword check for "cloud" might misclassify a weather query as a technical RAG query (or vice versa). An LLM router understands **intent** and context.
* **Robustness:** This allows the system to correctly handle ambiguous queries like *"I am feeling under the weather"* (which should go to RAG/General, not Weather API) or *"Tell me about the climate for investment"* (Financial RAG, not Weather).
* **Future-Proofing:** This design is easier to extend to 3+ tools (e.g., adding a "Calculator" or "Web Search") without writing complex, overlapping regex patterns.

### 2. Hybrid Retrieval Strategy
Pure vector search often fails on specific domain keywords. I implemented **Hybrid Retrieval**:
* **Dense (Qdrant):** Captures semantic meaning ("What is the atmosphere like?").
* **Sparse (BM25):** Captures exact keyword matches ("SECTION 3.2").
* **Reranking:** A Cross-Encoder filters the combined results to ensure the LLM only sees the most relevant context, reducing hallucinations.

### Project Structure


```text
├── app/
│   ├── evaluation/     # LangSmith tracing & evaluation wrappers
│   ├── graph/          # LangGraph nodes (Decision, Weather, RAG)
│   ├── rag/            # Retrieval logic (Embeddings, Ingestion, Qdrant)
│   ├── llm/            # LLM Client configuration
│   └── utils/          # API wrappers
├── data/               # Source PDFs
├── tests/              # Pytest unit tests
├── streamlit_app.py    # UI Entry point
├── Dockerfile          # Container config
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
