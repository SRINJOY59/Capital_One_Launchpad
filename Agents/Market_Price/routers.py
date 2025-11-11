from fastapi import APIRouter, HTTPException
import logging
from .agent import MarketPriceAgent
from .schemas import MarketPriceRequest, MarketPriceResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/marketprice", tags=["MarketPrice"])

_agent_instance = None

def get_agent() -> MarketPriceAgent:
    global _agent_instance
    if _agent_instance is None:
        try:
            _agent_instance = MarketPriceAgent()
            logger.info("Market price agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to initialize agent")
    return _agent_instance

@router.post("/analyze", response_model=MarketPriceResponse)
async def analyze_market_price(request: MarketPriceRequest):
    try:
        agent = get_agent()
        result = agent.get_market_analysis(request.query)
        return MarketPriceResponse(
            success=True,
            response=result
        )
    except Exception as e:
        logger.error(f"Market price analysis error: {str(e)}")
        return MarketPriceResponse(
            success=False,
            response=None,
            error=f"Failed to analyze market price: {str(e)}"
        )
