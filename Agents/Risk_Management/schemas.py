from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict

class RiskAssessmentRequest(BaseModel):
    query: str = Field(..., description="Query for agricultural risk analysis")

class RiskAssessmentResponse(BaseModel):
    success: bool = Field(..., description="Whether risk assessment was successful")
    risk_analysis: Optional[Any] = Field(None, description="Detailed risk analysis results")
    recommendations: Optional[List[str]] = Field(None, description="Risk mitigation recommendations")
    timestamp: Optional[str] = Field(None, description="Timestamp of analysis")
    error: Optional[str] = Field(None, description="Error message if assessment failed")
    recommendations: Optional[List[str]] = Field(None, description="Risk mitigation recommendations")
    timestamp: Optional[str] = Field(None, description="Timestamp of analysis")
    error: Optional[str] = Field(None, description="Error message if assessment failed")
