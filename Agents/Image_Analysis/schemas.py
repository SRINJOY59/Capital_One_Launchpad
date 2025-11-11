from pydantic import BaseModel, Field
from typing import Optional

class ImageAnalysisResponse(BaseModel):
    success: bool = Field(..., description="Whether image analysis was successful")
    description: Optional[str] = Field(None, description="Text description of the image")
    image_path: Optional[str] = Field(None, description="Path to the uploaded image")
    error: Optional[str] = Field(None, description="Error message if analysis failed")
