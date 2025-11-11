from fastapi import APIRouter, Query
from .agent import CropYieldAssistant
from .schemas import CropYieldRequest, CropYieldResponse

router = APIRouter(prefix = "/api/v1/agent", tags = ["Crop Yield"])
assistant = CropYieldAssistant()

@router.post("/predict", response_model=CropYieldResponse)
def predict_crop_yield(request: CropYieldRequest):
    result = assistant.respond(request.query)
    return CropYieldResponse(result=result)
