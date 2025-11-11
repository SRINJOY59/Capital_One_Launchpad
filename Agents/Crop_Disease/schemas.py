from pydantic import BaseModel, Field
from typing import Optional

class CropDiseaseDetectionResponse(BaseModel):
    success: bool = Field(..., description="Whether disease detection was successful")
    diseases: Optional[list[str]] = Field(None, description="Diagnosis and recommendations for crop disease")
    disease_probabilities: Optional[list[float]] = Field(None, description="Probabilities for each detected disease")
    symptoms: Optional[list[str]] = Field(None, description="Symptoms observed in the crop")
    Treatments: Optional[list[str]] = Field(None, description="Recommendations for treatment and prevention")
    prevention_tips: Optional[list[str]] = Field(None, description="Advice for prevention")
    image_path: Optional[str] = Field(None, description="Path to the uploaded image")
    error: Optional[str] = Field(None, description="Error message if detection failed")