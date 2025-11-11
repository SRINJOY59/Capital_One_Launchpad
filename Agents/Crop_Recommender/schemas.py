from pydantic import BaseModel
from typing import List

class CropRecommendationRequest(BaseModel):
    prompt: str

class CropRecommendationResponse(BaseModel):
    crop_names: List[str]
    confidence_scores: List[float]
    justifications: List[str]
