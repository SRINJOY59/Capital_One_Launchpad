from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uvicorn
import os
import tempfile
import time
from pathlib import Path
from Agents.Multi_Lingual.routers import router as multilingual_router
from Agents.Risk_Management.routers import router as risk_router
from Agents.Web_Scrapping.routers import router as webscrap_router
from Agents.Credit_Policy_Market.routers import router as creditpolicy_router
from Agents.Weather_forcast.routers import router as weather_router
from Agents.Crop_Disease.routers import router as crop_disease_router
from Agents.Market_Price.routers import router as market_price_router
from Agents.Image_Analysis.routers import router as image_analysis_router
# from Agents.Pest_prediction.routers import router as pest_prediction_router
from Agents.Crop_Recommender.routers import router as crop_recommender_router
from Agents.Crop_Yield.routers import router as crop_yield_router
from Agents.Location_Information.routers import router as location_information_router
from Agents.News.routers import router as news_router
from Agents.Risk_Modelling.routers import router as risk_router
from Agents.Personalisation.routers import router as personalisation_router
from Agents.Chart_Agent.routers import router as chart_agent_router
from Deep_Research.routers import router as deep_research_router
from Agents.Fertilizer_Recommender.routers import router as fertilizer_recommender_router
# from Tools.tool_apis_router import router as tool_apis_router

from workflow import run_workflow

app = FastAPI(
    title="Agricultural Agent API",
    description="API for multilingual agricultural assistance, analytics, and hybrid workflow processing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WorkflowRequestNormalQuery(BaseModel):
    query: str = Field(..., description="The agricultural query to process")
    mode: str = Field(default="rag", description="Initial processing mode: 'rag' or 'tooling'")

class WorkflowRequestImageQuery(BaseModel):
    query: str = Field(..., description="The agricultural query to process")
    image_path: str = Field(..., description="Path to image file")

class WorkflowResponse(BaseModel):
    answer: str
    answer_quality_grade: Dict[str, Any]
    is_answer_complete: bool
    final_mode: str
    switched_modes: bool
    is_image_query: bool
    chart_path: Optional[str] = None
    chart_extra_message: Optional[str] = None
    processing_time: Optional[float] = None

app.include_router(multilingual_router)
app.include_router(risk_router)
app.include_router(webscrap_router)
app.include_router(creditpolicy_router)
app.include_router(weather_router)
app.include_router(crop_disease_router)
app.include_router(market_price_router)
app.include_router(image_analysis_router)
# app.include_router(pest_prediction_router)
app.include_router(crop_recommender_router)
app.include_router(location_information_router)
app.include_router(news_router)
app.include_router(risk_router)
app.include_router(fertilizer_recommender_router)
app.include_router(deep_research_router)
app.include_router(crop_yield_router)
app.include_router(deep_research_router)
app.include_router(personalisation_router)
app.include_router(chart_agent_router)
# app.include_router(tool_apis_router)

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": "Agricultural AI API",
        "version": "1.0.0",
        "timestamp": time.time()
    }

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Agricultural AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.post("/api/v1/workflow/process", response_model=WorkflowResponse, tags=["Hybrid Workflow"])
async def process_workflow_query(request: WorkflowRequestNormalQuery):
    try:
        if request.mode.lower() not in ["rag", "tooling"]:
            raise HTTPException(
                status_code=400, 
                detail="Mode must be either 'rag' or 'tooling'"
            )

        start_time = time.time()
        result = run_workflow(
            query=request.query,
            mode=request.mode.lower(),
            image_path=None
        )
        end_time = time.time()
        processing_time = end_time - start_time

        if "answer_quality_grade" in result and hasattr(result["answer_quality_grade"], "dict"):
            result["answer_quality_grade"] = result["answer_quality_grade"].dict()
        elif "answer_quality_grade" in result and hasattr(result["answer_quality_grade"], "__dict__"):
            result["answer_quality_grade"] = dict(result["answer_quality_grade"].__dict__)

        result["processing_time"] = processing_time

        return WorkflowResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/v1/workflow/process-with-image", tags=["Hybrid Workflow"])
async def process_workflow_with_image(
    query: str,
    image: Optional[UploadFile] = File(None)
):
    temp_file_path = None
    try:
        image_path = None

        if image:
            allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            file_extension = Path(image.filename).suffix.lower()

            if file_extension not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
                )

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                    content = await image.read()
                    tmp.write(content)
                    temp_file_path = tmp.name
                    image_path = temp_file_path
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error saving uploaded image: {str(e)}")

        start_time = time.time()
        result = run_workflow(
            query=query,
            mode="tooling",
            image_path=image_path
        )
        end_time = time.time()
        processing_time = end_time - start_time

        if "answer_quality_grade" in result and hasattr(result["answer_quality_grade"], "dict"):
            result["answer_quality_grade"] = result["answer_quality_grade"].dict()
        elif "answer_quality_grade" in result and hasattr(result["answer_quality_grade"], "__dict__"):
            result["answer_quality_grade"] = dict(result["answer_quality_grade"].__dict__)

        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_err:
                print(f"Warning: Could not clean up temp file: {cleanup_err}")

        result["processing_time"] = processing_time

        return JSONResponse(content=result)

    except ValueError as e:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_err:
                print(f"Warning: Could not clean up temp file: {cleanup_err}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_err:
                print(f"Warning: Could not clean up temp file: {cleanup_err}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/v1/workflow/modes", tags=["Hybrid Workflow"])
async def get_workflow_modes():
    return {
        "available_modes": ["rag", "tooling"],
        "mode_descriptions": {
            "rag": "Retrieval-Augmented Generation using document search and synthesis",
            "tooling": "Agent-based processing using specialized agricultural tools"
        },
        "default_mode": "rag",
        "adaptive_switching": "Automatically switches modes based on answer quality assessment"
    }

@app.get("/api/v1/workflow/agents", tags=["Hybrid Workflow"])
async def get_available_agents():
    return {
        "available_agents": [
            "CropRecommenderAgent",
            "WeatherForecastAgent",
            "LocationAgriAssistant", 
            "NewsAgent",
            "CreditPolicyMarketAgent",
            "CropDiseaseDetectionAgent",
            "ImageAnalysisAgent",
            "MarketPriceAgent",
            "MultiLanguageTranslatorAgent",
            "PestPredictionAgent",
            "RiskManagementAgent",
            "WebScrapingAgent",
            "CropYieldAgent",
            "FertilizerRecommenderAgent",
            "ChartAgent"
        ],
        "agent_descriptions": {
            "CropRecommenderAgent": "Provides crop recommendations based on conditions",
            "WeatherForecastAgent": "Weather analysis and forecasting",
            "LocationAgriAssistant": "Location-specific agricultural information",
            "NewsAgent": "Agricultural news and market updates", 
            "CreditPolicyMarketAgent": "Credit policies and market analysis",
            "CropDiseaseDetectionAgent": "Disease detection and diagnosis",
            "ImageAnalysisAgent": "General image analysis capabilities",
            "MarketPriceAgent": "Market price analysis and trends",
            "MultiLanguageTranslatorAgent": "Multi-language translation services",
            "PestPredictionAgent": "Pest prediction and management",
            "RiskManagementAgent": "Agricultural risk assessment",
            "WebScrapingAgent": "Real-time data scraping",
            "CropYieldAgent": "Crop yield estimation and analysis",
            "FertilizerRecommenderAgent": "Fertilizer recommendations",
            "ChartAgent": "Data visualization and chart generation"
        },
        "routing": "Agents are automatically selected by RouterAgent based on query content"
    }

@app.get("/api/v1/workflow/test-queries", tags=["Hybrid Workflow"])
async def get_test_queries():
    return {
        "text_queries": [
            "Hello how are you?",
            "Estimate crop yield for wheat in Punjab in winter of 2025",
            "How can farmers manage pest outbreaks in cotton fields?",
            "What is the market price trend for wheat in India?",
            "How to prevent fungal diseases in tomato crops?",
            "Create a chart showing corn price trends over the last year",
            "Recommend fertilizers for rice cultivation in monsoon season"
        ],
        "image_queries": [
            "Analyze this crop disease",
            "Check for pests in this image",
            "Identify the crop type in this field",
            "Assess soil quality from this image"
        ],
        "chart_queries": [
            "Visualize rainfall patterns in agricultural regions",
            "Show seasonal commodity price fluctuations",
            "Compare crop yields across different states",
            "Graph fertilizer price trends over time"
        ]
    }

@app.post("/api/v1/workflow/batch-process", tags=["Hybrid Workflow"])
async def batch_process_queries(queries: list[str], mode: str = "rag"):
    if mode.lower() not in ["rag", "tooling"]:
        raise HTTPException(
            status_code=400,
            detail="Mode must be either 'rag' or 'tooling'"
        )
    
    results = []
    total_start_time = time.time()
    
    for i, query in enumerate(queries):
        try:
            start_time = time.time()
            result = run_workflow(
                query=query,
                mode=mode.lower(),
                image_path=None
            )
            end_time = time.time()
            
            if "answer_quality_grade" in result and hasattr(result["answer_quality_grade"], "dict"):
                result["answer_quality_grade"] = result["answer_quality_grade"].dict()
            elif "answer_quality_grade" in result and hasattr(result["answer_quality_grade"], "__dict__"):
                result["answer_quality_grade"] = dict(result["answer_quality_grade"].__dict__)
            
            result["processing_time"] = end_time - start_time
            result["query_index"] = i
            result["original_query"] = query
            
            results.append(result)
            
        except Exception as e:
            results.append({
                "query_index": i,
                "original_query": query,
                "error": str(e),
                "processing_time": 0
            })
    
    total_end_time = time.time()
    
    return {
        "batch_results": results,
        "total_queries": len(queries),
        "successful_queries": len([r for r in results if "error" not in r]),
        "failed_queries": len([r for r in results if "error" in r]),
        "total_processing_time": total_end_time - total_start_time,
        "average_processing_time": (total_end_time - total_start_time) / len(queries)
    }

@app.get("/api/v1/workflow/status", tags=["Hybrid Workflow"])
async def get_workflow_status():
    return {
        "workflow_status": "active",
        "supported_features": [
            "Hybrid RAG-Agent processing",
            "Adaptive mode switching", 
            "Quality-driven answer generation",
            "Image analysis capabilities",
            "Chart generation",
            "Multi-language support",
            "Batch processing",
            "Real-time data integration"
        ],
        "current_capabilities": {
            "rag_mode": "Available",
            "tooling_mode": "Available", 
            "image_processing": "Available",
            "chart_generation": "Available",
            "offline_mode": "Available with HF Model",
            "quality_grading": "Available"
        }
    }

@app.get("/api-info", tags=["Information"])
async def api_info():
    return {
        "api_version": "1.0.0",
        "workflow_features": {
            "hybrid_processing": "Combines RAG and agent-based approaches",
            "adaptive_mode_switching": "Automatically switches between RAG and tooling based on answer quality",
            "quality_grading": "Evaluates answer completeness and switches strategies if needed",
            "query_rewriting": "Improves query formulation for better results",
            "image_processing": "Handles image-based agricultural queries",
            "chart_generation": "Creates data visualizations with real market data",
            "parallel_execution": "Runs multiple agents concurrently for efficiency",
            "batch_processing": "Process multiple queries simultaneously"
        },
        "supported_agents": [
            "Crop Recommendation", "Weather Forecasting", "Location Agriculture",
            "Agricultural News", "Credit Policy & Market", "Crop Disease Detection", 
            "Image Analysis", "Market Pricing", "Multi-language Translation",
            "Pest Prediction", "Risk Management", "Web Scraping", "Crop Yield Estimation",
            "Chart Generation", "Fertilizer Recommendation"
        ],
        "workflow_modes": {
            "rag": "Retrieval-Augmented Generation using document search and synthesis",
            "tooling": "Agent-based processing using specialized agricultural tools"
        },
        "new_features": {
            "chart_generation": "Real-time data visualization with market insights",
            "enhanced_image_processing": "Improved crop disease and pest detection",
            "batch_processing": "Process multiple queries efficiently",
            "workflow_status_monitoring": "Track system capabilities and performance"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )