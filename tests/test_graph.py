"""
Test LangGraph Agent
-------------------
Tests agent routing logic.
"""

from langchain_core.documents import Document  
from app.graph.graph import agent_graph

def test_weather_query_routes_to_weather_node(mocker):
    """
    Ensure weather-related queries are routed correctly.
    """
    
    # return a list of Documents, matching the real weather_node
    mock_weather_doc = Document(
        page_content="It is sunny in Delhi.", 
        metadata={"source": "weather_api"}
    )

    mocker.patch(
        "app.graph.weather_node.weather_node",
        return_value={
            "answer": "It is sunny.",
            "source": "weather_api",
            "context": [mock_weather_doc] 
        }
    )

    state = {
        "query": "What is the weather in Delhi?",
        "answer": None,
        "source": None,
        "context": None
    }

    # Note: This runs the REAL decision_node (using the LLM).
    result = agent_graph.invoke(state)

    assert result["source"] == "weather_api"
    assert isinstance(result["answer"], str)
    assert isinstance(result["context"], list) 
    assert isinstance(result["context"][0], Document)


def test_rag_query_routes_to_rag_node(mocker):
    """
    Ensure document-related queries are routed to RAG.
    """

    mocker.patch(
        "app.graph.rag_node.rag_node",
        return_value={
            "answer": "Hybrid RAG combines retrieval and generation.",
            "source": "rag",
            "context": [] 
        }
    )

    state = {
        "query": "Explain Hybrid RAG architecture",
        "answer": None,
        "source": None,
        "context": None
    }

    result = agent_graph.invoke(state)

    assert result["source"] == "rag"
    assert isinstance(result["answer"], str)