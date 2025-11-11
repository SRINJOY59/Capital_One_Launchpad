from pydantic import BaseModel, Field
from typing import Optional

class ChartRequest(BaseModel):
    query: str = Field(..., description="Agricultural query for chart generation")
    
class ChartResponse(BaseModel):
    extra_message: str = Field(..., description="Insights and recommendations based on data analysis")
    code: str = Field(..., description="Python code that generates the visualization")
    image_path: str = Field(..., description="Full path where the chart image is saved")
    success: bool = Field(default=True, description="Indicates if chart generation was successful")
    error_message: Optional[str] = Field(None, description="Error message if generation failed")
