from fastapi import APIRouter
from pydantic import BaseModel
from .agent import LocationAgriAssistant

router = APIRouter(prefix="/api/v1/location", tags=["LocationAgri"])
assistant = LocationAgriAssistant()

class LocationQuery(BaseModel):
    prompt: str

@router.post("/location-agri/query")
async def location_agri_query(query: LocationQuery):
    try:
        response = assistant.respond(query.prompt)
        return {"success": True, "response": response}
    except Exception as e:
        return {"success": False, "error": str(e)}
