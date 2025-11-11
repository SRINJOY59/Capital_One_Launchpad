from fastapi import APIRouter, UploadFile, File, HTTPException, Form
import os
import shutil
from .agent import CropDiseaseAgent
from .schemas import CropDiseaseDetectionResponse

router = APIRouter(prefix="/api/v1/cropdisease", tags=["CropDiseaseDetection"])

_agent_instance = None

def get_agent() -> CropDiseaseAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = CropDiseaseAgent()
    return _agent_instance

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/detect", response_model=CropDiseaseDetectionResponse)
async def detect_disease(
    image: UploadFile = File(None),
    query: str = Form("describe the diseases")
):
    try:
        agent = get_agent()
        temp_path = None
        
        if image:
            file_ext = os.path.splitext(image.filename)[-1]
            temp_path = os.path.join(UPLOAD_DIR, image.filename)
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
        
        result = agent.analyze_disease(query=query, image_path=temp_path)
        
        diseases = result.diseases if result.diseases else []
        disease_probabilities = result.disease_probabilities if result.disease_probabilities else []
        symptoms = result.symptoms if result.symptoms else []
        treatments = result.Treatments if result.Treatments else []
        prevention_tips = result.prevention_tips if result.prevention_tips else []
        
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        
        return CropDiseaseDetectionResponse(
            success=True,
            diseases=diseases,
            disease_probabilities=disease_probabilities,
            symptoms=symptoms,
            Treatments=treatments,
            prevention_tips=prevention_tips,
            image_path=temp_path
        )
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        
        return CropDiseaseDetectionResponse(
            success=False,
            diseases=[],
            disease_probabilities=[],
            symptoms=[],
            Treatments=[],
            prevention_tips=[],
            image_path=None,
            error=f"Failed to detect crop disease: {str(e)}"
        )