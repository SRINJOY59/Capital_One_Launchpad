from pydantic import BaseModel
from typing import List

class AssistantRequest(BaseModel):
    user_location: str
    preferred_language: str
    crops: List[str]
    total_land_area: int
    season : str
    farming_type : str
    irrigation : str
    budget : str
    experience : str

class AssistantResponse(BaseModel):
    answer: str
