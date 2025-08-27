from fastapi import APIRouter, HTTPException
from .agent import AgriculturalChartAgent
from .schemas import ChartRequest, ChartResponse

router = APIRouter(prefix="/chart", tags=["Agricultural Charts"])
agent = AgriculturalChartAgent()

@router.post("/generate", response_model=ChartResponse)
async def generate_chart(request: ChartRequest):
    try:
        response = agent.generate_response(request.query)
        
        if not response:
            raise HTTPException(status_code=500, detail="Failed to generate chart")
        
        return ChartResponse(
            extra_message=response.extra_message,
            code=response.code,
            image_path=response.imagekit_url,
            success=True
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Agricultural Chart Agent"}
