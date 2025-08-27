from pydantic import BaseModel
from typing import Optional, List

class RiskAnalysisRequest(BaseModel):
    geography: str
    objectives: Optional[List[str]] = None
    mode: str = "collaborate"
    model_id: str = "gemini-2.0-flash"

class RiskAnalysisResponse(BaseModel):
    analysis: str
