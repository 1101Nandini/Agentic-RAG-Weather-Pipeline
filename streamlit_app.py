"""
Streamlit UI
------------
A simple chat interface to interact with the agentic AI pipeline.
"""

import streamlit as st
from app.graph.graph import agent_graph
from app.evaluation.langsmith_eval import trace_agent_response
from app.rag.ingest import ingest_documents

ingest_documents()

st.set_page_config(
    page_title="Agentic Hybrid RAG Demo",
    layout="centered"
)

st.title("🤖 LangGraph Agentic RAG & Weather Assistant")

st.markdown(
    """
Ask me:
- 🌦️ Weather-related questions (e.g., *What's the weather in Delhi?*)
- 📄 Questions from the knowledge base
"""
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_query = st.chat_input("Type your question here...")

if user_query:
    # Display user message
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )
    with st.chat_message("user"):
        st.markdown(user_query)

    # Invoke agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            state = {
                "query": user_query,
                "answer": None,
                "source": None,
                "context": None
            }

            result_state = agent_graph.invoke(state)

            # Trace with LangSmith
            trace_agent_response(result_state)

            answer = result_state.get("answer", "No answer generated.")
            source = result_state.get("source", "unknown")

            st.markdown(answer)
            st.caption(f"Source: `{source}`")

    # Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
