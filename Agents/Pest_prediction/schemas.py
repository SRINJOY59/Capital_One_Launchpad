from pydantic import BaseModel, Field
from typing import Optional, List

class PestPredictionRequest(BaseModel):
    query: str = Field(..., description="Pest prediction query")
    image_path: Optional[str] = Field(None, description="Path to pest image for identification")

class PestPredictionResponse(BaseModel):
    success: bool = Field(..., description="Whether the agent processed the query successfully")
    possible_pest_names: Optional[List[str]] = Field(None, description="List of possible pest names identified")
    description: Optional[str] = Field(None, description="Description of pest symptoms and impact")
    pesticide_recommendation: Optional[str] = Field(None, description="Recommended pesticides or treatments")
    error: Optional[str] = Field(None, description="Error message if processing failed")
