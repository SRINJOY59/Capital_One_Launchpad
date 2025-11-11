import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://weather.googleapis.com/v1/forecast/days:lookup"


def get_google_weather_forecast(latitude: float, longitude: float) -> dict:
    api_key = os.getenv("GOOGLE_WEATHER_API_KEY")
    params = {
        "location.latitude": latitude,
        "location.longitude": longitude,
        "key": api_key,
        "days": 7
    }

    response = requests.get(API_URL, params=params)
    print(response)
    if response.status_code != 200:
        return {"error": f"Failed to fetch data, status code: {response.status_code}", "response": response.json()}

    data = response.json()
    forecast = []
    
    daily_forecasts = data.get("forecastDays", [])
    
    for day in daily_forecasts:
        date_obj = day.get("displayDate", {})
        date_str = f"{date_obj.get('year')}-{date_obj.get('month'):02d}-{date_obj.get('day'):02d}"

        # Daytime and Nighttime forecasts are separated
        daytime_forecast = day.get("daytimeForecast", {})
        nighttime_forecast = day.get("nighttimeForecast", {})

        # Temperature
        max_temp = day.get("maxTemperature", {}).get("degrees")
        min_temp = day.get("minTemperature", {}).get("degrees")

        # Precipitation and Thunderstorm
        precip_prob_day = daytime_forecast.get("precipitation", {}).get("probability", {}).get("percent")
        precip_sum_day = daytime_forecast.get("precipitation", {}).get("qpf", {}).get("quantity")
        thunderstorm_prob_day = daytime_forecast.get("thunderstormProbability")

        # Humidity
        humidity_day = daytime_forecast.get("relativeHumidity")
        humidity_night = nighttime_forecast.get("relativeHumidity")
        
        # Wind
        wind_speed_day = daytime_forecast.get("wind", {}).get("speed", {}).get("value")
        wind_gust_day = daytime_forecast.get("wind", {}).get("gust", {}).get("value")

        # Sun Events
        sunrise_time = day.get("sunEvents", {}).get("sunriseTime")
        sunset_time = day.get("sunEvents", {}).get("sunsetTime")

        # Cloud Cover
        cloud_cover_day = daytime_forecast.get("cloudCover")
        
        # Heat Index
        max_heat_index = day.get("maxHeatIndex", {}).get("degrees")

        forecast.append({
            "date": date_str,
            "temperatures_celsius": {
                "max": max_temp,
                "min": min_temp
            },
            "humidity_percent": {
                "daytime": humidity_day,
                "nighttime": humidity_night
            },
            "precipitation": {
                "sum_mm": precip_sum_day,
                "probability_percent": precip_prob_day,
            },
            "thunderstorm_probability_percent": thunderstorm_prob_day,
            "wind_kmh": {
                "speed_max": wind_speed_day,
                "gust_max": wind_gust_day
            },
            "cloud_cover_percent": cloud_cover_day,
            "uv_index_max": daytime_forecast.get("uvIndex"),
            "max_heat_index_celsius": max_heat_index,
            "sun_events": {
                "sunrise": sunrise_time,
                "sunset": sunset_time
            }
        })

    return {
        "location": {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": data.get("timeZone", {}).get("id")
        },
        "forecast_days": forecast
    }

if __name__ == "__main__":
    lat, lon = 28.6139, 77.2090  # New Delhi, India

    forecast_data = get_google_weather_forecast(lat, lon)
    print(json.dumps(forecast_data, indent=4))
