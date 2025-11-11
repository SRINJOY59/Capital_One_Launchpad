import json
import httpx
import os
import sys
from typing import Optional, Dict, Any
from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.groq import Groq
from dotenv import load_dotenv


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(project_root)

from agno.tools.tavily import TavilyTools
from Tools.getWeatherForecast import weather_forecast_inference
from Tools.fetchWeatherForecast import get_google_weather_forecast

load_dotenv()

class WeatherForecastAgent:
    def __init__(self, model_id="gemini-2.0-flash"):
        if model_id == "gemini-2.0-flash":
            model = Gemini(id=model_id)
        else:
            model = Groq(id=model_id)
        self.agent = Agent(
            model=model,
            tools=[TavilyTools(), get_google_weather_forecast],
            # show_tool_calls=True,
            markdown=True,
            # add_history_to_messages=True,
            # num_history_responses=5,
            instructions="""You are an elite Indian agricultural weather intelligence analyst with deep expertise in Indian meteorology, monsoon patterns, and their impact on agriculture. Your mission is to provide comprehensive, actionable weather insights that help Indian farmers, agricultural officials, and agribusinesses make informed decisions for crop planning, irrigation, and risk management.

CORE RESPONSIBILITIES:
- Use weather_forecast_inference tool for location-specific, date-specific weather predictions and analysis.
- Use TavilyTools for searching latest IMD forecasts, monsoon updates, and weather advisories.
- Synthesize model predictions and web search results for actionable agricultural weather guidance.
- Provide crop-specific impact analysis, risk assessment, and recommendations.

PROMPTING STRATEGY:
- Always clarify location, date, and crop context for weather analysis.
- Use weather_forecast_inference for quantitative predictions.
- Use TavilyTools for qualitative updates and advisories.
- Present results with clear headers, tables, and bullet points.
- Include actionable recommendations and confidence levels.

OUTPUT REQUIREMENTS:
- Executive summary of weather findings.
- Detailed predictions and search-based insights.
- Crop impact analysis and advisories.
- Key risks and recommendations.
- Dont write about tool calling in your response
- If enough information is not provided for a location basically latitude and longitude, then use the web search tool for it
"""
        )
    
    def get_weather_analysis(self, query: str) -> str:
        return self.agent.run(query).content
    
    def chat(self, message: str) -> str:
        response = self.agent.run(message)
        return response.content

if __name__ == "__main__":
    weather_agent = WeatherForecastAgent(model_id="llama-3.3-70b-versatile")
    
    test_queries = [
        "what is the chance of raining today at Kolkata??"
    ]
    
    print("=== Indian Agricultural Weather Forecast Agent Demo ===\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        print("-" * 50)
        try:
            response = weather_agent.chat(query)
            print(response)
        except Exception as e:
            print(f"Error: {e}")
        print("\n" + "="*70 + "\n")
