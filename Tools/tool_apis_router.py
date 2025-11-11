from fastapi import APIRouter, UploadFile, File, Form
from market_inform_policy_capture import MarketInformPolicyCapture
from web_scrapper import scrape_agri_prices, scrape_policy_updates, scrape_links
from translation_tool import MultiLanguageTranslator
from risk_management import get_agricultural_risk_metrics
from pest_prediction import detect_pests
from getCropYield import crop_yield_inference
from getCropRecommendation import get_crop_recommendation
from fetchWeatherForecast import get_google_weather_forecast
from fetchMarketPrice import fetch_market_price
from fertilizer_inference import FertilizerRecommendationInference
from crop_disease_detection import detect_crop_disease
import tempfile

router = APIRouter()

market_capture_tool = MarketInformPolicyCapture()
translator = MultiLanguageTranslator()

@router.get("/api/v1/creditpolicy/comprehensive-analysis")
async def comprehensive_analysis():
    try:
        result = market_capture_tool.run_comprehensive_analysis()
        return {"success": True, "result": str(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/api/v1/webscrapper/agri-prices")
async def agri_prices(url: str, table_selector: str = "table"):
    try:
        data = scrape_agri_prices(url, table_selector)
        return {"success": True, "data": data}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/api/v1/webscrapper/policy-updates")
async def policy_updates(url: str, selector: str = ".policy-update"):
    try:
        updates = scrape_policy_updates(url, selector)
        return {"success": True, "updates": updates}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/api/v1/webscrapper/links")
async def links(url: str, selector: str = "a"):
    try:
        links = scrape_links(url, selector)
        return {"success": True, "links": links}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/api/v1/translate/text")
async def translate_text(text: str, source_lang: str = "auto", target_lang: str = "en"):
    try:
        result = translator.translate_robust(text, source_lang, target_lang)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/api/v1/translate/batch")
async def batch_translate(texts: list, source_lang: str = "auto", target_lang: str = "en"):
    try:
        results = translator.batch_translate(texts, source_lang, target_lang)
        return {"success": True, "results": results}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/api/v1/riskmanagement/agricultural-risk")
async def agricultural_risk(primary_commodity: str):
    try:
        result = get_agricultural_risk_metrics(primary_commodity)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/api/v1/pest-prediction")
async def pest_prediction(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        pests = detect_pests(tmp_path)
        return {"success": True, "detected_pests": pests}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/api/v1/crop-yield/predict")
async def crop_yield_predict(
    state_name: str,
    district_name: str,
    crop_year: int,
    season: str,
    crop: str,
    temperature: float,
    humidity: float,
    soil_moisture: float,
    area: float,
    model_type: str = "stacked_2"
):
    try:
        result_json = crop_yield_inference(
            state_name=state_name,
            district_name=district_name,
            crop_year=crop_year,
            season=season,
            crop=crop,
            temperature=temperature,
            humidity=humidity,
            soil_moisture=soil_moisture,
            area=area,
            model_type=model_type
        )
        return result_json
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/api/v1/crop-recommendation")
async def crop_recommendation(
    N: float,
    P: float,
    K: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float,
    model_type: str = "stacked"
):
    try:
        result_json = get_crop_recommendation(
            N=N,
            P=P,
            K=K,
            temperature=temperature,
            humidity=humidity,
            ph=ph,
            rainfall=rainfall,
            model_type=model_type,
            return_format='json'
        )
        return result_json
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/api/v1/weather/forecast-tool")
async def weather_forecast(
    latitude: float,
    longitude: float
):
    try:
        result = get_google_weather_forecast(latitude, longitude)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/api/v1/market-price")
async def market_price(state_name: str = "Karnataka"):
    try:
        result = fetch_market_price(state_name)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/api/v1/fertilizer/recommendation")
async def fertilizer_recommendation(
    temperature: float,
    humidity: float,
    moisture: float,
    soil_type: str,
    crop_type: str,
    nitrogen: float,
    potassium: float,
    phosphorous: float
):
    try:
        predictor = FertilizerRecommendationInference()
        errors = predictor.validate_inputs(
            temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous
        )
        if errors:
            return {"success": False, "validation_errors": errors}
        result = predictor.predict(
            temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous
        )
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/api/v1/crop-disease/detect")
async def crop_disease_detect(file: UploadFile = File(...)):
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        result = detect_crop_disease(tmp_path)
        return {"success": True, "diseases": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
