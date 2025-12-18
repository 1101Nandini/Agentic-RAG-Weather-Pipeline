"""
Weather API Utility
-------------------
This module provides a clean wrapper around the OpenWeatherMap API.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()


class WeatherAPIError(Exception):
    """Custom exception for Weather API errors."""
    pass


def fetch_weather(city: str) -> dict:
    """
    Fetches real-time weather data for a given city.

    Args:
        city (str): City name

    Returns:
        dict: Structured weather information
    """

    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise WeatherAPIError("OPENWEATHER_API_KEY not set")

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        raise WeatherAPIError(f"Weather API request failed: {e}")

    data = response.json()

    return {
        "city": city,
        "temperature_celsius": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "description": data["weather"][0]["description"]
    }
