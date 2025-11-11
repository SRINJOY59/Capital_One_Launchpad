from fastapi import APIRouter, HTTPException
from .agent import FertilizerRecommendationAgent
from .schemas import FertilizerQuery, FertilizerOutput

router = APIRouter(prefix = "/api/v1", tags = ["Fertilizer"])

@router.post("/recommend", response_model=FertilizerOutput)
def recommend_fertilizer(query: FertilizerQuery):
    agent = FertilizerRecommendationAgent()
    result = agent.recommend_fertilizer(query.query)
    try:
        return FertilizerOutput(
            primary_fertilizer=result.primary_fertilizer,
            confidence_score=result.confidence_score,
            alternative_fertilizers=result.alternative_fertilizers,
            application_rate=result.application_rate,
            timing=result.timing,
            cost_estimate=result.cost_estimate,
            benefits=result.benefits,
            precautions=result.precautions
        )
    except Exception as e:
        return FertilizerOutput(
            primary_fertilizer="NPK 10-10-10",
            confidence_score=0.5,
            alternative_fertilizers=["Urea", "DAP"],
            application_rate="200 kg per hectare",
            timing="Apply during sowing season",
            cost_estimate="$300-400 per hectare",
            benefits=["Improves crop yield", "Better soil health"],
            precautions=["Avoid over-application", "Check soil pH"]
        )
    