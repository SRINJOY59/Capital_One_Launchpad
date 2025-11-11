from fastapi import APIRouter, HTTPException
import logging
from .agent import AgriculturalRiskAnalysisAgent
from .schemas import RiskAssessmentRequest, RiskAssessmentResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/risk", tags=["RiskManagement"])

_agent_instance = None

def get_agent() -> AgriculturalRiskAnalysisAgent:
    global _agent_instance
    if _agent_instance is None:
        try:
            _agent_instance = AgriculturalRiskAnalysisAgent()
            logger.info("Risk management agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to initialize agent")
    return _agent_instance

@router.post("/analyze", response_model=RiskAssessmentResponse)
async def analyze_risk(request: RiskAssessmentRequest):
    try:
        agent = get_agent()
        result = agent.analyze_risk(request.query)
        return RiskAssessmentResponse(
            success=True,
            risk_analysis=result,
            recommendations=None,
            timestamp=None
        )
    except Exception as e:
        logger.error(f"Risk analysis error: {str(e)}")
        return RiskAssessmentResponse(
            success=False,
            risk_analysis=None,
            recommendations=None,
            timestamp=None,
            error=f"Failed to analyze risk: {str(e)}"
        )
        
