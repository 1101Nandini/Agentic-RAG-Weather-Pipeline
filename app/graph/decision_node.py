"""
Decision Node (LLM-Based)
-------------------------
Routes queries to specialized tools using an LLM classifier.
"""

from typing import Dict
from langchain_core.prompts import PromptTemplate
from app.llm.llm_client import get_llm

ROUTER_PROMPT = PromptTemplate(
    template="""<|im_start|>system
You are an intelligent query router. Your job is to classify the user's intent into exactly one of two categories: "weather" or "rag".

Categories:
1. "weather": strictly for queries asking about current weather conditions, temperature, forecast, rain, or humidity in specific locations.
2. "rag": for EVERYTHING else. This includes general questions, definitions, summaries, specific document queries, or any topic not related to weather.

Output ONLY one word: "weather" or "rag".
<|im_end|>
<|im_start|>user
Query: {query}
<|im_end|>
<|im_start|>assistant
""",
    input_variables=["query"]
)

def decision_node(state: Dict) -> Dict:
    query = state.get("query", "")
    llm = get_llm()
    
    # Run the classification
    response = llm.invoke(ROUTER_PROMPT.format(query=query))
    
    # Normalize and clean the output
    route_raw = response.strip().lower()
    
    # Default to 'rag' for safety, only switch if explicitly 'weather'
    if "weather" in route_raw:
        final_route = "weather"
    else:
        final_route = "rag"

    return {**state, "route": final_route}