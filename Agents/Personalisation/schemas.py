from pydantic import BaseModel
from typing import List

class AssistantRequest(BaseModel):
    user_location: str
    preferred_language: str
    crops: List[str]
    total_land_area: int

class AssistantResponse(BaseModel):
    answer: str
