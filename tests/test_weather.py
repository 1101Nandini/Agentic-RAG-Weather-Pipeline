"""
Test Weather API
----------------
Tests the weather API utility using mocking to avoid real HTTP calls.
"""

import pytest
import requests
from app.utils.weather_api import fetch_weather, WeatherAPIError


def test_fetch_weather_success(mocker):
    """
    Test successful weather API response.
    """

    mock_response = {
        "main": {"temp": 25, "humidity": 60},
        "weather": [{"description": "clear sky"}],
    }

    mock_get = mocker.patch("requests.get")
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = mock_response

    result = fetch_weather("Delhi")

    assert result["city"] == "Delhi"
    assert result["temperature_celsius"] == 25
    assert result["humidity"] == 60
    assert result["description"] == "clear sky"


def test_fetch_weather_api_error(mocker):
    """
    Test API failure handling.
    """

    mock_get = mocker.patch("requests.get")
    mock_get.side_effect = requests.RequestException("API error")

    with pytest.raises(WeatherAPIError):
        fetch_weather("Delhi")
