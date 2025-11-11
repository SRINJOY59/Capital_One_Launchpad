from fastapi import APIRouter, HTTPException
import logging
from .agent import CreditPolicyMarketAgent
from .schemas import CreditPolicyMarketRequest, CreditPolicyMarketResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/creditpolicy", tags=["CreditPolicyMarket"])

_agent_instance = None

def get_agent() -> CreditPolicyMarketAgent:
    global _agent_instance
    if _agent_instance is None:
        try:
            _agent_instance = CreditPolicyMarketAgent()
            logger.info("Credit Policy Market agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to initialize agent")
    return _agent_instance

@router.post("/analyze", response_model=CreditPolicyMarketResponse)
async def analyze_credit_policy(request: CreditPolicyMarketRequest):
    try:
        agent = get_agent()
        result = agent.respond_to_query(request.query)
        return CreditPolicyMarketResponse(
            success=True,
            response=result
        )
    except Exception as e:
        logger.error(f"Credit policy analysis error: {str(e)}")
        return CreditPolicyMarketResponse(
            success=False,
            response=None,
            error=f"Failed to analyze credit policy: {str(e)}"
        )
