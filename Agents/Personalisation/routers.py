from fastapi import APIRouter, HTTPException
from .agent import PersonalizedAssistant
from .schemas import AssistantRequest, AssistantResponse

router = APIRouter(prefix="/api/v1", tags=["Personalization"])

@router.post("/personalised-advice", response_model=AssistantResponse)
def get_personalised_advice(request: AssistantRequest):
    try:
        assistant = PersonalizedAssistant(
            user_location=request.user_location,
            preferred_language=request.preferred_language,
            crops=request.crops,
            total_land_area=request.total_land_area,
            season=request.season,
            farming_type=request.farming_type,
            irrigation=request.irrigation,
            budget=request.budget,
            experience=request.experience
        )
        answer = assistant.run()
        return AssistantResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
