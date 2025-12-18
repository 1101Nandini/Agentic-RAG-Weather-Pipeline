"""
LangGraph 
--------------------
This module defines the agentic workflow using LangGraph.

Flow:
User Query
   ↓
Decision Node
   ├── Weather Node → Final Answer
   └── RAG Node     → Final Answer
"""

from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

from app.graph.decision_node import decision_node
from app.graph.weather_node import weather_node
from app.graph.rag_node import rag_node


class AgentState(TypedDict):
    query: str
    route: Optional[str]
    answer: Optional[str]
    source: Optional[str]
    context: Optional[List[Document]]


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("decision", decision_node)
    graph.add_node("weather", weather_node)
    graph.add_node("rag", rag_node)

    graph.set_entry_point("decision")

    # routing based on state["route"]
    graph.add_conditional_edges(
        "decision",
        lambda state: state["route"],
        {
            "weather": "weather",
            "rag": "rag",
        }
    )

    graph.add_edge("weather", END)
    graph.add_edge("rag", END)

    return graph.compile()


agent_graph = build_graph()
