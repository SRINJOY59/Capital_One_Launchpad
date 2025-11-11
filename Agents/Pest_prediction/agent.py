import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(project_root)
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.tavily import TavilyTools
# from Tools.pest_prediction import detect_pests
from dotenv import load_dotenv
from agno.media import Image
from pathlib import Path
from pydantic import BaseModel
from typing import List

load_dotenv()

class PestPredictionOutput(BaseModel):
    possible_pest_names: List[str]
    description: str
    pesticide_recommendation: str

class PestPredictionAgent:
    def __init__(self, model_id="gemini-2.0-flash"):
        self.agent = Agent(
            model=Gemini(id=model_id),
            markdown=True,
            # show_tool_calls=True,
            # add_history_to_messages=True,
            # num_history_responses=5,
            output_schema=PestPredictionOutput,
            tools=[
                # detect_pests,
                TavilyTools()
            ],
            instructions="""
You are an expert agricultural pest management specialist with advanced AI-powered pest identification capabilities.

PROMPTING STRATEGY:
- When an image is provided, use the detect_pests tool to identify all possible pest names present in the image.
- Provide structured output with possible_pest_names, description of symptoms and impact, and pesticide_recommendation.
- Always include confidence levels for pest identification if available.
- Give actionable pesticide recommendations and monitoring advice.
- Use clear, concise language suitable for farmers and agronomists.

OUTPUT REQUIREMENTS:
- possible_pest_names: List of all pest names detected in the image.
- description: Description of pest symptoms and impact.
- pesticide_recommendation: Recommended pesticides or treatments.
"""
        )

    def respond(self, query, image_path=None):
        if image_path and os.path.exists(image_path):
            image = Image(filepath=Path(image_path))
            response = self.agent.run(query, images=[image]).content
        else:
            response = self.agent.run(query).content
        return response
        
        
if __name__ == "__main__":
    agent = PestPredictionAgent()
    query = "Identify pests in this image and provide recommendations."
    image_path = "../../Dataset/pest/test/bollworm/jpg_0.jpg" 
    result = agent.respond(query, image_path)
    print(result)  
