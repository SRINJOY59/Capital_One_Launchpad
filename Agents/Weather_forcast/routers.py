from fastapi import APIRouter, HTTPException
import logging
from .agent import WeatherForecastAgent
from .schemas import WeatherForecastRequest, WeatherForecastResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/weather", tags=["WeatherForecast"])

_agent_instance = None

def get_agent() -> WeatherForecastAgent:
    global _agent_instance
    if _agent_instance is None:
        try:
            _agent_instance = WeatherForecastAgent()
            logger.info("Weather forecast agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to initialize agent")
    return _agent_instance

@router.post("/forecast", response_model=WeatherForecastResponse)
async def forecast_weather(request: WeatherForecastRequest):
    try:
        agent = get_agent()
        result = agent.get_weather_analysis(request.query)
        return WeatherForecastResponse(
            success=True,
            response=result
        )
    except Exception as e:
        logger.error(f"Weather forecast error: {str(e)}")
        return WeatherForecastResponse(
            success=False,
            response=None,
            error=f"Failed to forecast weather: {str(e)}"
        )
