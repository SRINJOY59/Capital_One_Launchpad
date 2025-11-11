from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import logging
from .agent import PestPredictionAgent
from .schemas import PestPredictionResponse
import tempfile
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pest", tags=["PestPrediction"])

_agent_instance = None

def get_agent() -> PestPredictionAgent:
    global _agent_instance
    if _agent_instance is None:
        try:
            _agent_instance = PestPredictionAgent()
            logger.info("Pest prediction agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to initialize agent")
    return _agent_instance

@router.post("/predict", response_model=PestPredictionResponse)
def predict_pest(query: str = Form(...), image: UploadFile = File(None)):
    try:
        image_path = None
        if image:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                shutil.copyfileobj(image.file, tmp)
                image_path = tmp.name
        agent = get_agent()
        result = agent.respond(query, image_path)
        if hasattr(result, "dict"):
            result = result.dict()
        if isinstance(result, dict):
            return PestPredictionResponse(
                success=True,
                possible_pest_names=result.get("possible_pest_names"),
                description=result.get("description"),
                pesticide_recommendation=result.get("pesticide_recommendation"),
                error=None
            )
        if isinstance(result, str):
            possible_pest_names = None
            description = None
            pesticide_recommendation = None
            if "possible_pest_names=" in result:
                try:
                    parts = result.split("description=")
                    pest_part = parts[0].split("possible_pest_names=")[-1].strip().strip("'").strip("[]")
                    possible_pest_names = [p.strip().strip("'") for p in pest_part.split(",") if p.strip()]
                    desc_part = parts[1].split("pesticide_recommendation=")[0].strip().strip("'")
                    description = desc_part
                    pesticide_part = parts[1].split("pesticide_recommendation=")[-1].strip().strip("'")
                    pesticide_recommendation = pesticide_part
                except Exception:
                    description = result
            else:
                description = result
            return PestPredictionResponse(
                success=True,
                possible_pest_names=possible_pest_names,
                description=description,
                pesticide_recommendation=pesticide_recommendation,
                error=None
            )
        return PestPredictionResponse(
            success=True,
            possible_pest_names=None,
            description=str(result),
            pesticide_recommendation=None,
            error=None
        )
    except Exception as e:
        logger.error(f"Pest prediction error: {str(e)}")
        return PestPredictionResponse(
            success=False,
            possible_pest_names=None,
            description=None,
            pesticide_recommendation=None,
            error=f"Failed to predict pest: {str(e)}"
        )
