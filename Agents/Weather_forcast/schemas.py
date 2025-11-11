from pydantic import BaseModel, Field
from typing import Optional, Any

class WeatherForecastRequest(BaseModel):
    query: str = Field(..., description="Weather forecast query for agriculture")

class WeatherForecastResponse(BaseModel):
    success: bool = Field(..., description="Whether the agent processed the query successfully")
    response: Optional[str] = Field(None, description="Agent's weather forecast response")
    error: Optional[str] = Field(None, description="Error message if processing failed")
