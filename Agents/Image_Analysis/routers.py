from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import shutil
from .agent import ImageAgent
from .schemas import ImageAnalysisResponse

router = APIRouter(prefix="/api/v1/image", tags=["ImageAnalysis"])

_agent_instance = None

def get_agent() -> ImageAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ImageAgent()
    return _agent_instance

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/describe", response_model=ImageAnalysisResponse)
async def describe_image(image: UploadFile = File(...)):
    try:
        file_ext = os.path.splitext(image.filename)[-1]
        temp_path = os.path.join(UPLOAD_DIR, image.filename)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        agent = get_agent()
        description = agent.describe_image(temp_path)
        return ImageAnalysisResponse(
            success=True,
            description=description,
            image_path=temp_path
        )
    except Exception as e:
        return ImageAnalysisResponse(
            success=False,
            description=None,
            image_path=None,
            error=f"Failed to analyze image: {str(e)}"
        )
