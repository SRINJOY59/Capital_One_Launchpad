from pydantic import BaseModel, Field
from typing import Optional

class MarketPriceRequest(BaseModel):
    query: str = Field(..., description="Query for market price agent")

class MarketPriceResponse(BaseModel):
    success: bool = Field(..., description="Whether the agent processed the query successfully")
    response: Optional[str] = Field(None, description="Agent's market price response")
    error: Optional[str] = Field(None, description="Error message if processing failed")
