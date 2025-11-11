from fastapi import APIRouter, HTTPException
from .agent import RiskMitigationOrchestrator
from .schemas import RiskAnalysisRequest, RiskAnalysisResponse

router = APIRouter(prefix="/api/v1", tags = ["Risk Analysis"])

@router.post("/risk-analysis", response_model=RiskAnalysisResponse)
def get_risk_analysis(request: RiskAnalysisRequest):
    try:
        orchestrator = RiskMitigationOrchestrator(
            mode=request.mode,
            model_id=request.model_id
        )
        output = orchestrator.run(
            geography=request.geography,
            objectives=request.objectives
        )
        return RiskAnalysisResponse(analysis=output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
