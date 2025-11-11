from fastapi import APIRouter
from pydantic import BaseModel
from .agent import NewsAgent

router = APIRouter(prefix="/api/v1/agent", tags=["news"])  # Fixed prefix format
news_agent = NewsAgent()

class NewsQuery(BaseModel):
    query: str

@router.post("/agri-news")
async def agri_news_endpoint(request: NewsQuery):
    try:
        response = news_agent.get_agri_news(request.query)
        return {"success": True, "response": response}
    except Exception as e:
        return {"success": False, "error": str(e)}
