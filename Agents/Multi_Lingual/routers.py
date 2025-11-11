from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
from .agent import MultiLingualAgent
from .schemas import AgricultureQueryRequest, AgricultureQueryResponse, TranslationRequest, TranslationResponse, HealthCheckResponse, ErrorResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agriculture", tags=["Agriculture"])

_agent_instance = None

def get_agent() -> MultiLingualAgent:
    global _agent_instance
    if _agent_instance is None:
        try:
            _agent_instance = MultiLingualAgent()
            logger.info("Multilingual agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to initialize agent")
    return _agent_instance

@router.post("/respond", response_model=AgricultureQueryResponse)
async def respond_query(request: AgricultureQueryRequest):
    try:
        agent = get_agent()
        response_content = agent.respond(request.query)
        return AgricultureQueryResponse(
            success=True,
            response=response_content,
            detected_language=None,
            response_language=str(request.preferred_response_lang or "auto"),
            processing_steps=None,
            error=None
        )
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        return AgricultureQueryResponse(
            success=False,
            response=None,
            detected_language=None,
            response_language=str(request.preferred_response_lang or "auto"),
            processing_steps=None,
            error=f"Failed to process query: {str(e)}"
        )
    
@router.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Translate text to the specified target language using the multilingual agent
    """
    try:
        agent = get_agent()
        print(request.source_lang)
        print(request.target_lang)
        # Use your agent's translation capability
        translated_text = agent.translate_text(
            text=request.text,
            source_lang=getattr(request, 'source_language', 'auto'),
            target_lang=request.target_lang
        )
        
        # Check if translation failed (your agent returns error message in the string)
        if translated_text.startswith("Translation failed:"):
            return TranslationResponse(
                success=False,
                translated_text=None,
                source_language=getattr(request, 'source_language', 'auto'),
                target_language=request.target_lang,
                error=translated_text
            )
        print(translated_text)

        return TranslationResponse(
            success=True,
            translated_text=translated_text,
            source_language=getattr(request, 'source_language', 'auto'),
            target_language=request.target_lang,
            error=None
        )
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return TranslationResponse(
            success=False,
            translated_text=None,
            source_language=getattr(request, 'source_language', 'auto'),
            target_language=request.target_lang,
            error=f"Translation failed: {str(e)}"
        )



@router.get("/health", response_model=HealthCheckResponse)
async def health():
    return HealthCheckResponse(
        status="ok",
        service="MultiLingualAgent",
        version="1.0.0",
        supported_languages=["auto", "en", "bn", "te", "hi", "ta"]
    )
