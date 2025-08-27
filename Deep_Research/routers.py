from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any
from datetime import datetime
import logging

from .orchastrator import AgriculturalWorkflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/deep-research", tags=["Deep Research"])

workflow_instance = None

def get_workflow_instance() -> AgriculturalWorkflow:
    """Dependency to get workflow instance"""
    global workflow_instance
    if workflow_instance is None:
        workflow_instance = AgriculturalWorkflow(max_iterations=3)
    return workflow_instance

# Request Model
class AgriculturalRequest(BaseModel):
    query: str = Field(
        ..., 
        description="Your agricultural question or request", 
        min_length=10, 
        max_length=1000,
        example="I want to grow tomatoes in my farm, need fertilizer recommendations and market prices"
    )
    response_format: Optional[Literal["simple", "detailed", "executive"]] = Field(
        default="simple", 
        description="Format of the response - simple for quick answers, detailed for comprehensive analysis, executive for summary"
    )
    max_iterations: Optional[int] = Field(
        default=3, 
        description="Maximum refinement iterations for better results", 
        ge=1, 
        le=5
    )

# Response Model
class AgriculturalResponse(BaseModel):
    success: bool = Field(..., description="Request success status")
    query: str = Field(..., description="Original agricultural query")
    response: str = Field(..., description="Agricultural guidance and recommendations")
    execution_time_seconds: float = Field(..., description="Time taken to process the request")
    response_format: str = Field(..., description="Format used for the response")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional execution metadata")

@router.post("/ask", response_model=AgriculturalResponse)
async def get_agricultural_guidance(
    request: AgriculturalRequest,
    workflow: AgriculturalWorkflow = Depends(get_workflow_instance)
):
    """
    Get comprehensive agricultural guidance and recommendations.
    
    **This single endpoint handles all your agricultural queries including:**
    - Crop recommendations and planning
    - Fertilizer and soil management advice
    - Weather and climate considerations
    - Market prices and economic analysis
    - Pest and disease management
    - Risk assessment and mitigation
    - Government schemes and financial assistance
    - And much more agricultural guidance
    
    **Parameters:**
    - **query**: Your detailed agricultural question or scenario
    - **response_format**: 
        - `simple`: Quick, actionable recommendations (default)
        - `detailed`: Comprehensive analysis with step-by-step guidance
        - `executive`: High-level summary with key insights
    - **max_iterations**: How many times the system should refine the answer for better quality (1-5)
    
    **Example queries:**
    - "I have 5 acres of land in Karnataka, what crops should I grow during monsoon season?"
    - "My wheat crop is showing yellow leaves, what could be the problem and solution?"
    - "What are the current market prices for rice and what's the profit margin?"
    - "I need a complete farming plan for organic vegetable farming"
    """
    
    try:
        start_time = datetime.now()
        logger.info(f"Processing agricultural query: {request.query[:100]}...")
        
        # Set max iterations on workflow instance
        workflow.max_iterations = request.max_iterations
        
        # Process based on response format
        if request.response_format == "simple":
            response_text = workflow.get_simple_answer(request.query)
            metadata = None
            
        elif request.response_format == "executive":
            response_text = workflow.get_executive_summary(request.query)
            
            # Get basic metrics for executive format
            final_state = workflow.execute_workflow(request.query)
            metadata = {
                "total_agents_used": len(final_state.get("final_results", [])),
                "successful_responses": len([r for r in final_state.get("final_results", []) if r.grade == "yes"]),
                "tools_utilized": final_state.get("research_plan", {}).tools_list if final_state.get("research_plan") else [],
                "execution_id": final_state.get("execution_id", "unknown")
            }
            
        else:  # detailed format
            response_text = workflow.execute_workflow_as_string(request.query)
            
            # Get detailed metadata
            final_state = workflow.execute_workflow(request.query)
            metadata = {
                "execution_id": final_state.get("execution_id", "unknown"),
                "total_agents": len(final_state.get("final_results", [])),
                "successful_agents": len([r for r in final_state.get("final_results", []) if r.status == "success"]),
                "high_quality_responses": len([r for r in final_state.get("final_results", []) if r.grade == "yes"]),
                "iterations_completed": final_state.get("current_iteration", 0),
                "tools_used": final_state.get("research_plan", {}).tools_list if final_state.get("research_plan") else [],
                "success_rate": f"{(len([r for r in final_state.get('final_results', []) if r.grade == 'yes']) / len(final_state.get('final_results', [])) * 100):.1f}%" if final_state.get("final_results") else "0%"
            }
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        logger.info(f"Successfully processed query in {execution_time:.2f} seconds")
        
        return AgriculturalResponse(
            success=True,
            query=request.query,
            response=response_text,
            execution_time_seconds=execution_time,
            response_format=request.response_format,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Error processing agricultural query: {str(e)}")
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        return AgriculturalResponse(
            success=False,
            query=request.query,
            response=f"Sorry, I encountered an error while processing your agricultural query: {str(e)}. Please try rephrasing your question or contact support if the issue persists.",
            execution_time_seconds=execution_time,
            response_format=request.response_format,
            metadata={"error": str(e), "error_type": type(e).__name__}
        )

