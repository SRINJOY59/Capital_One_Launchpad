import os
import sys
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.google_maps import GoogleMapTools
from agno.tools.tavily import TavilyTools
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional, List

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(project_root)

from Tools.fertilizer_inference import FertilizerRecommendationInference
load_dotenv()

class FertilizerOutput(BaseModel):
    primary_fertilizer: str = "NPK 10-10-10"
    confidence_score: float = 0.5
    alternative_fertilizers: List[str] = ["Urea", "DAP"]
    application_rate: str = "200 kg per hectare"
    timing: str = "Apply during sowing season"
    cost_estimate: str = "$300-400 per hectare"
    benefits: List[str] = ["Improves crop yield", "Better soil health"]
    precautions: List[str] = ["Avoid over-application", "Check soil pH"]

def recommend_fertilizer_tool(temperature: float, humidity: float, moisture: float, 
                             soil_type: str, crop_type: str, nitrogen: float, 
                             potassium: float, phosphorous: float):
    """ML model tool for fertilizer recommendation"""
    try:
        predictor = FertilizerRecommendationInference()
        result = predictor.predict(temperature, humidity, moisture, soil_type, 
                                 crop_type, nitrogen, potassium, phosphorous)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

class FertilizerRecommendationAgent:
    def __init__(self, model_id="gemini-2.0-flash"):
        self.agent = Agent(
            model=Gemini(id=model_id),
            markdown=True,
            debug_mode=False,
            # show_tool_calls=True,
            # add_history_to_messages=True,
            # num_history_responses=5,
            tools=[recommend_fertilizer_tool, GoogleMapTools(), TavilyTools()],
            output_schema=FertilizerOutput,
            instructions="""
You are a fertilizer recommendation expert. Analyze user queries and provide detailed, practical fertilizer advice.

PROCESS:
1. Extract parameters from query: temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous
2. If location mentioned, use GoogleMapTools for weather data
3. If missing soil/crop data, use TavilyTools for regional information  
4. Call recommend_fertilizer_tool when you have sufficient parameters
5. Use TavilyTools for current prices and best practices
6. Provide comprehensive recommendations

PARAMETER RANGES:
- temperature: 20-40°C
- humidity: 30-75%
- moisture: 25-65%
- soil_type: Sandy/Loamy/Black/Red/Clayey
- nitrogen, potassium, phosphorous: 0-50

OUTPUT FORMAT (populate ALL fields):
{
  "primary_fertilizer": "NPK 10-26-26 (specific fertilizer name)",
  "confidence_score": 0.85,
  "alternative_fertilizers": ["Urea", "DAP", "Organic compost"],
  "application_rate": "200-250 kg per hectare in 2-3 splits",
  "timing": "First application at sowing, second at 30 days after sowing",
  "cost_estimate": "$300-400 per hectare including labor",
  "benefits": ["Increases yield by 15-20%", "Improves soil nitrogen", "Better root development"],
  "precautions": ["Do not apply during rain", "Maintain soil moisture", "Test soil pH regularly"]
}

Make recommendations specific to the crop, soil, and region. Include practical application details, realistic cost estimates, and actionable advice.
"""
        )

    def recommend_fertilizer(self, query: str):
        result = self.agent.run(f"Provide detailed fertilizer recommendations for: {query}")
        
        if hasattr(result, 'content'):
            response = result.content
        else:
            response = result
            
        return response
    
if __name__ == "__main__":
    query = "I need fertilizer for wheat in Punjab, soil test shows low nitrogen"
    agent = FertilizerRecommendationAgent()
    recommendations = agent.recommend_fertilizer(query)
    
    print(f"Primary Fertilizer: {recommendations.primary_fertilizer}")
    print(f"Confidence: {recommendations.confidence_score}")
    print(f"Alternatives: {recommendations.alternative_fertilizers}")
    print(f"Application Rate: {recommendations.application_rate}")
    print(f"Timing: {recommendations.timing}")
    print(f"Cost: {recommendations.cost_estimate}")
    print(f"Benefits: {recommendations.benefits}")
    print(f"Precautions: {recommendations.precautions}")