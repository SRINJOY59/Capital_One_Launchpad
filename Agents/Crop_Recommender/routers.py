from fastapi import APIRouter
from .agent import CropRecommenderAgent
from .schemas import CropRecommendationRequest, CropRecommendationResponse

router = APIRouter(prefix = "/api/v1/crop-recommender", tags = ["Crop Recommender"])
agent = CropRecommenderAgent()

@router.post("/crop-recommendation", response_model=CropRecommendationResponse)
async def crop_recommendation_endpoint(request: CropRecommendationRequest):
    result = agent.respond(request.prompt)
    try:
        return CropRecommendationResponse(
            crop_names=result.crop_names,
            confidence_scores=result.confidence_scores,
            justifications=result.justifications
        )
    except Exception as e:
        return CropRecommendationResponse(
            crop_names=[],
            confidence_scores=[],
            justifications=[]
        )
