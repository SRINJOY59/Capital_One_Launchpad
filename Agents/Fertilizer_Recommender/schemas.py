from pydantic import BaseModel
from typing import List

class FertilizerQuery(BaseModel):
    query: str

class FertilizerOutput(BaseModel):
    primary_fertilizer: str = "NPK 10-10-10"
    confidence_score: float = 0.5
    alternative_fertilizers: List[str] = ["Urea", "DAP"]
    application_rate: str = "200 kg per hectare"
    timing: str = "Apply during sowing season"
    cost_estimate: str = "$300-400 per hectare"
    benefits: List[str] = ["Improves crop yield", "Better soil health"]
    precautions: List[str] = ["Avoid over-application", "Check soil pH"]
