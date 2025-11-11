from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
from .agent import AgriculturalWebScrappingAgent
from .schemas import WebScrappingRequest, WebScrappingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/webscrap", tags=["WebScrapping"])

_agent_instance = None

def get_agent() -> AgriculturalWebScrappingAgent:
    global _agent_instance
    if _agent_instance is None:
        try:
            _agent_instance = AgriculturalWebScrappingAgent()
            logger.info("Web scrapping agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to initialize agent")
    return _agent_instance

@router.post("/scrape", response_model=WebScrappingResponse)
async def scrape(request: WebScrappingRequest):
    try:
        agent = get_agent()
        result = agent.scrape(request.query)
        return WebScrappingResponse(
            success=True,
            data=result,
            sources=None
        )
    except Exception as e:
        logger.error(f"Web scraping error: {str(e)}")
        return WebScrappingResponse(
            success=False,
            data=None,
            sources=None,
            error=f"Failed to scrape: {str(e)}"
        )
