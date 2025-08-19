import os
import sys
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.tavily import TavilyTools
from pydantic import BaseModel
from agno.tools.google_maps import GoogleMapTools
from dotenv import load_dotenv
from typing import Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(project_root)

from Tools.fetchWeatherForecast import get_google_weather_forecast
from Tools.getCropRecommendation import get_consensus_prediction, get_crop_recommendation, get_all_model_predictions

load_dotenv()

class CropRecommendation(BaseModel):
   crop_names: Optional[list[str]]
   confidence_scores: Optional[list[float]]
   justifications: Optional[list[str]]

class CropRecommenderAgent:
   def __init__(self):
       self.agent = Agent(
           model=Gemini(id="gemini-2.0-flash"),
           tools=[TavilyTools(), GoogleMapTools(), get_google_weather_forecast, get_all_model_predictions, get_consensus_prediction, get_crop_recommendation],
           instructions="""
You are an expert crop recommendation agent specializing in Indian agriculture. Your expertise covers soil analysis, climate assessment, weather patterns, and agricultural economics.

CORE RESPONSIBILITIES:
- Analyze soil parameters (N, P, K, pH, organic matter, moisture)
- Evaluate climate conditions (temperature, humidity, rainfall patterns)
- Consider geographical and topographical factors
- Assess market conditions and crop profitability
- Provide season-specific recommendations
- Use the default model type (Random forest) for calling the tool 
- Account for water availability and irrigation requirements

ANALYSIS PROCESS:
1. If location is provided: fetch weather data, soil characteristics, and regional agricultural patterns
2. If soil parameters are given: cross-reference with crop requirements and local conditions
3. If season/timing is mentioned: align recommendations with planting calendars
4. Always incorporate current weather forecasts and seasonal projections
5. Consider market prices, demand trends, and potential risks

TOOL USAGE STRATEGY:
- Use weather tools for current conditions and forecasting
- Use mapping tools for geographical and climatic data
- Use search tools for market trends and agricultural practices
- Use crop prediction models for data-driven recommendations
- Combine multiple model outputs for consensus recommendations

OUTPUT REQUIREMENTS:
- Provide exactly 3 crop recommendations ranked by suitability
- Each recommendation must include:
 - Specific crop name
 - Confidence score (0.0-1.0)
 - Comprehensive justification covering soil suitability, climate match, water requirements, market potential, seasonal timing, risk factors, and expected yield
- Justifications should be detailed (minimum 50 words each) and include specific data points
- Consider crop rotation benefits and sustainable practices
- Address potential challenges and mitigation strategies
- Include actionable next steps for farmers

RESPONSE STYLE:
- Be authoritative yet accessible to farmers
- Use specific agricultural terminology appropriately
- Include quantitative data where available
- Focus on practical, implementable advice
- Avoid mentioning internal tool operations
""",
           show_tool_calls=False,
           markdown=True,
           response_model=CropRecommendation,
       )

   def respond(self, prompt: str) -> CropRecommendation:
       return self.agent.run(prompt).content

if __name__ == "__main__":
   agent = CropRecommenderAgent()
   
   test_queries = [
       "What type of crop recommendations are you seeking for?"
       "I have 5 acres in Punjab with sandy loam soil, pH 7.2, moderate rainfall. What crops should I plant this Kharif season?",
       "My soil has high nitrogen (280 ppm), low phosphorus (15 ppm), medium potassium (145 ppm). Location: Uttar Pradesh. Best crops for summer cultivation?",
       "I'm in Maharashtra, Pune district. Monsoon is expected to be normal. What cash crops give good returns with minimal water?",
       "Small farmer in Tamil Nadu with 2 acres. Soil pH is 6.8, good organic content. Looking for crops that mature in 90-120 days.",
       "Bihar location, alluvial soil, good irrigation facility. Want high-value crops for export market. Current season: post-monsoon."
   ]
   
   for i, query in enumerate(test_queries, 1):
       print(f"\n=== Test Query {i} ===")
       print(f"Query: {query}")
       print("\nResponse:")
       result = agent.respond(query)
       print(f"Crops: {result.crop_names}")
       print(f"Confidence: {result.confidence_scores}")
       for j, justification in enumerate(result.justifications):
           print(f"Justification {j+1}: {justification}")
       print("-" * 80)