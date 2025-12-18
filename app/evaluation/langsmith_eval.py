"""
LangSmith Evaluation & Tracing
------------------------------
"""

from typing import Dict, Any
from langsmith import traceable


@traceable(name="agent_response")
def trace_agent_response(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "query": state.get("query"),
        "answer": state.get("answer"),
        "source": state.get("source"),
        "num_context_docs": len(state.get("context", []))
        if state.get("context") else 0,
        "context_preview": [
            doc.page_content[:200]
            for doc in (state.get("context") or [])
        ],
    }

