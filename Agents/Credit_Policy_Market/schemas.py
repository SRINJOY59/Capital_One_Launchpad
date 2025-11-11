from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class CreditPolicyMarketRequest(BaseModel):
    query: str = Field(..., description="Query for credit policy and market agent")

class CreditPolicyMarketResponse(BaseModel):
    success: bool = Field(..., description="Whether the agent processed the query successfully")
    response: Optional[str] = Field(None, description="Agent's response")
    error: Optional[str] = Field(None, description="Error message if processing failed")
