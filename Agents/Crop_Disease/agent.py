import os
import sys
from agno.agent import Agent
from agno.media import Image
from pathlib import Path
from agno.models.google import Gemini
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(project_root)

# from Tools.crop_disease_detection import detect_crop_disease
from agno.tools.tavily import TavilyTools
load_dotenv()


class CropDiseaseOutput(BaseModel):
    diseases: list[str] = []
    disease_probabilities: list[float] = []
    symptoms: list[str] = []
    Treatments: list[str] = []
    prevention_tips: list[str] = []

class CropDiseaseAgent:
    def __init__(self, model_id="gemini-2.0-flash"):
        self.agent = Agent(
            model=Gemini(id=model_id),
            markdown=True,
            debug_mode=False,
            # show_tool_calls=True,
            # add_history_to_messages=True,
            # num_history_responses=5,
            tools=[TavilyTools()],
            output_schema=CropDiseaseOutput,
            instructions="""
You are an advanced crop disease analysis agent. Your task is to analyze crop images for disease symptoms and provide a clear diagnosis and actionable recommendations.

CRITICAL: You MUST return EXACTLY this JSON structure:
{
    "diseases": ["Disease1", "Disease2", "Disease3"],
    "disease_probabilities": [0.85, 0.70, 0.60],
    "symptoms": ["symptom1", "symptom2", "symptom3"],
    "Treatments": ["treatment1", "treatment2", "treatment3"],
    "prevention_tips": ["tip1", "tip2", "tip3"]
}

RULES:
1. ALWAYS provide at least 3 items in each list
2. disease_probabilities should be decimals between 0.0 and 1.0
3. If no image provided, use general crop disease information
4. Each prevention tip should be maximum 10 words
5. Make treatments specific and actionable

EXAMPLE OUTPUT FOR HEALTHY CROP:
{
    "diseases": ["Healthy", "No disease detected", "Normal growth"],
    "disease_probabilities": [0.95, 0.90, 0.85],
    "symptoms": ["Green healthy leaves", "Normal growth pattern", "No visible damage"],
    "Treatments": ["Continue regular care", "Monitor for changes", "Maintain current practices"],
    "prevention_tips": ["Regular watering schedule", "Balanced fertilization program", "Pest monitoring routine"]
}

When analyzing, use the crop_disease_detection tool if an image is provided, then supplement with your knowledge and TavilyTools for additional context.
"""
        )

    def analyze_disease(self, query: str, image_path=None):
        if image_path and os.path.exists(image_path):
            print("Image path exists")
            image = Image(filepath=Path(image_path))
            prompt = f"Analyze this crop image for disease symptoms and provide diagnosis with structured output: {query}"
            result = self.agent.run(prompt, images=[image])
        else:
            prompt = f"No image provided. Analyze based on context only. Set diseases and disease_probabilities to empty lists, but provide symptoms, treatments, and prevention tips for: {query}"
            result = self.agent.run(prompt)
        
        # Check if result has content attribute
        if hasattr(result, 'content'):
            response = result.content
        else:
            response = result
            
        #print(f"Agent CropDiseaseDetectionAgent Response: {response}")
        
        # Validate the response has the required fields
        if (hasattr(response, "diseases") and hasattr(response, "disease_probabilities") and 
            hasattr(response, "symptoms") and hasattr(response, "Treatments") and 
            hasattr(response, "prevention_tips")):
            
            # Check if all critical fields are None/empty
            if (not response.diseases and not response.disease_probabilities and 
                not response.symptoms and not response.Treatments and not response.prevention_tips):
                print("Warning: All outputs are empty. Creating default response.")
                # Create a default response
                response = CropDiseaseOutput(
                    diseases=["Unable to determine"],
                    disease_probabilities=[0.0],
                    symptoms=["Image analysis failed"],
                    Treatments=["Consult local agricultural expert"],
                    prevention_tips=["Regular crop monitoring recommended"]
                )
        
        return response

if __name__ == "__main__":
    agent = CropDiseaseAgent()
    # Use absolute path to ensure image is found
    image_path = os.path.join(project_root, "uploads", "Screenshot 2025-08-16 at 12.22.35 PM.png")
    print(f"Looking for image at: {image_path}")
    print(f"Image exists: {os.path.exists(image_path)}")
    
    result = agent.analyze_disease(query="analyze this crop for diseases", image_path=image_path)
    print("Diseases:", result.diseases)
    print("Disease probabilities:", result.disease_probabilities)
    print("Symptoms:", result.symptoms)
    print("Treatments:", result.Treatments)
    print("Prevention tips:", result.prevention_tips)

    query = "What are the common diseases affecting wheat crops?"
    result = agent.analyze_disease(query=query)
    print("Disease names:", result.disease_name)
    print("Disease probabilities:", result.disease_probability)
    print("Symptoms:", result.symptoms)
    print("Treatments:", result.Treatments)
    print("Prevention tips:", result.prevention_tips)