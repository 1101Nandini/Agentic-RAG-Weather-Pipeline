"""
Weather Node
------------
"""

from typing import Dict
from langchain_core.documents import Document 
from app.utils.weather_api import fetch_weather

def weather_node(state: Dict) -> Dict:
    """
    Executes the weather tool and formats a response.
    """
    query = state.get("query", "")
    city = extract_city_from_query(query)

    # 1. Fetch Data
    weather = fetch_weather(city)

    # 2. Format Answer
    answer = (
        f"The current weather in {weather['city']} is "
        f"{weather['description']}, with a temperature of "
        f"{weather['temperature_celsius']}°C and humidity "
        f"around {weather['humidity']}%."
    )

    # 3. Create a Document object for consistency
    # This prevents the 'AttributeError: str has no attribute page_content'
    weather_doc = Document(
        page_content=answer,
        metadata={
            "source": "weather_api",
            "city": weather["city"],
            "temperature": weather["temperature_celsius"],
            "humidity": weather["humidity"]
        }
    )

    # 4. Update State
    state["answer"] = answer
    state["source"] = "weather_api"
    state["context"] = [weather_doc]  

    return state


def extract_city_from_query(query: str) -> str:
    """
    Naive city extraction from query.
    Sufficient for demo purposes.
    """
    tokens = query.lower().split()
    if "in" in tokens:
        # Grab the word immediately after "in"
        # e.g. "weather in Tokyo" -> "Tokyo"
        try:
            return tokens[tokens.index("in") + 1].strip("?.,")
        except IndexError:
            pass # Fallback if "in" is the last word
            
    return tokens[-1].strip("?.,")