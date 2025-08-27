import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
import logging
from datetime import datetime
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from Tools.getWeatherForecast import weather_forecast_inference
from Tools.fetchWeatherForecast import get_google_weather_forecast
from Tools.fertilizer_inference import FertilizerRecommendationInference
from Tools.fetchMarketPrice import fetch_market_price
from Tools.getCropRecommendation import get_crop_recommendation
from Tools.getCropYield import crop_yield_inference
from Tools.pest_prediction import detect_pests
from Tools.crop_disease_detection import detect_crop_disease
# from Tools.webSearch import web_search_tool
from Tools.web_scrapper import scrape_agri_prices, scrape_links, scrape_policy_updates
from agno.tools.google_maps import GoogleMapTools
from Tools.risk_management import get_agricultural_risk_metrics
# from Tools.agriculturalNews import agricultural_news_tool
# from Tools.farmCreditPolicy import farm_credit_tool
# from Tools.translation import translation_tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskExecutionResult(BaseModel):
    task_id: str = Field(..., description="Task identifier")
    task_description: str = Field(..., description="Task description")
    tools_used: List[str] = Field(..., description="Tools used for this task")
    execution_status: str = Field(..., description="Execution status (success/failed/partial)")
    result_content: str = Field(..., description="Generated content from task execution")
    tool_outputs: Dict[str, str] = Field(default_factory=dict, description="Individual tool outputs")

class ExecutionPlan(BaseModel):
    execution_id: str = Field(..., description="Execution identifier")
    total_tasks: int = Field(..., description="Total number of tasks")
    completed_tasks: int = Field(default=0, description="Number of completed tasks")
    task_results: List[TaskExecutionResult] = Field(default_factory=list, description="Task execution results")
    overall_status: str = Field(default="pending", description="Overall execution status")

class TaskExecutorAgent:
    def __init__(self):
        load_dotenv()
        
        self.available_tools = {
            "Fertilizer Recommendation Tool": FertilizerRecommendationInference,
            # "Web Search Tool": web_search_tool,
            "Market Price Tool": fetch_market_price,
            "Weather Forecast Tool": get_google_weather_forecast,
            "Crop Recommendation Tool": get_crop_recommendation,
            "Crop Yield Prediction Tool": crop_yield_inference,
            "Pest Detection Tool": detect_pests,
            # "Translation Tool": translation_tool,
            "Crop Disease Tool": detect_crop_disease,
            # "Google Maps Location Tool": maps_location_tool,
            "Web Scrapper Tool": scrape_agri_prices,
            "Risk Management Tool": get_agricultural_risk_metrics,
            # "Agricultural News Tool": agricultural_news_tool,
            # "Farm Credit Policy Tool": farm_credit_tool
        }
        
        self.executor = Agent(
            name="Agricultural Task Executor",
            model=Gemini(id="gemini-2.0-flash"),
            instructions=self._get_executor_instructions(),
        )

    def _get_executor_instructions(self) -> str:
        return """You are an Agricultural Task Executor Agent. Your role is to:

1. Execute agricultural tasks using assigned tools
2. Coordinate multiple tools when needed for a single task
3. Generate comprehensive content based on tool outputs
4. Provide actionable recommendations and insights
5. Ensure all aspects of the task are addressed

When executing tasks:
- Use all assigned tools for the task
- Synthesize information from multiple tool outputs
- Provide practical, actionable advice
- Include specific recommendations with quantities, timings, and methods
- Consider regional and contextual factors
- Ensure content is farmer-friendly and easy to understand

Generate detailed, informative content that directly addresses the task requirements."""

    def execute_tasks(self, research_plan, user_query: str = "") -> ExecutionPlan:
        execution_plan = ExecutionPlan(
            execution_id=f"EX{datetime.now().strftime('%Y%m%d%H%M%S')}",
            total_tasks=len(research_plan.tasks),
            overall_status="in_progress"
        )
        
        # for task in research_plan.tasks:
        #     task_id = f"TASK_{id+1:03d}"
            
        for idx, task in enumerate(research_plan.tasks, start=1):
            try:
                task.task_id = f"TASK_{idx:03d}"  
                logger.info(f"Executing {task.task_id}: {task.description}")
                
                tool_output = ""
                if task.tool_assignment in self.available_tools:
                    tool_function = self.available_tools[task.tool_assignment]
                    try:
                        tool_output = tool_function(task.description, user_query)
                    except Exception as tool_error:
                        logger.error(f"Error in tool {task.tool_assignment}: {str(tool_error)}")
                        tool_output = f"Error executing {task.tool_assignment}: {str(tool_error)}"
                    logger.info(f"Tool {task.tool_assignment} executed successfully")
                else:
                    logger.warning(f"Tool {task.tool_assignment} not available")
                    tool_output = f"Tool {task.tool_assignment} is not available"
                
                content = self._generate_task_content(task.description, {task.tool_assignment: tool_output}, user_query)
                
                result = TaskExecutionResult(
                    task_id=task.task_id,
                    task_description=task.description,
                    tools_used=[task.tool_assignment],
                    execution_status="success",
                    result_content=content,
                    tool_outputs={task.tool_assignment: tool_output}
                )
                
                execution_plan.task_results.append(result)
                execution_plan.completed_tasks += 1
                
            except Exception as e:
                logger.error(f"Error executing {task.task_id}: {str(e)}")
                
                result = TaskExecutionResult(
                    task_id=task.task_id,
                    task_description=task.description,
                    tools_used=[task.tool_assignment],
                    execution_status="failed",
                    result_content=f"Task execution failed: {str(e)}",
                    tool_outputs={}
                )
                
                execution_plan.task_results.append(result)
        
        execution_plan.overall_status = "completed" if execution_plan.completed_tasks == execution_plan.total_tasks else "partial"
        
        return execution_plan

    def _generate_task_content(self, task_description: str, tool_outputs: Dict[str, str], user_query: str) -> str:
        try:
            synthesis_prompt = f"""
            Task: {task_description}
            Original Query: {user_query}
            
            Tool Outputs:
            {chr(10).join([f"{tool}: {output}" for tool, output in tool_outputs.items()])}
            
            Synthesize this information into comprehensive, actionable content that addresses the task.
            Provide specific recommendations, quantities, methods, and practical advice.
            Make the content farmer-friendly and easy to understand.
            """
            
            response = self.executor.run(synthesis_prompt)
            
            if hasattr(response, 'content'):
                return str(response.content)
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            
            synthesized_content = f"Task: {task_description}\n\n"
            for tool, output in tool_outputs.items():
                synthesized_content += f"{tool} Results:\n{output}\n\n"
            
            return synthesized_content

    def display_execution_results(self, execution_plan: ExecutionPlan):
        print(f"\n{'='*80}")
        print(f"TASK EXECUTION RESULTS")
        print(f"{'='*80}")
        print(f"Execution ID: {execution_plan.execution_id}")
        print(f"Status: {execution_plan.overall_status.upper()}")
        print(f"Tasks: {execution_plan.completed_tasks}/{execution_plan.total_tasks}")
        
        for result in execution_plan.task_results:
            print(f"\n{'-'*60}")
            print(f"Task ID: {result.task_id}")
            print(f"Description: {result.task_description}")
            print(f"Tools Used: {', '.join(result.tools_used)}")
            print(f"Status: {result.execution_status}")
            print(f"\nGenerated Content:")
            print(f"{result.result_content}")

def main():
    from datetime import datetime
    
    # Import the planner agent
    from planner_agent import AgriculturalPlanningAgent
    
    # Initialize both agents
    planner = AgriculturalPlanningAgent()
    executor = TaskExecutorAgent()
    
    # Sample query
    sample_query = "I want to grow tomatoes and need fertilizer recommendations and market prices"
    
    # Get plan from planner agent
    research_plan = planner.create_plan(sample_query)
    
    # Display the plan
    planner.display_plan(research_plan)
    
    # Execute the plan
    results = executor.execute_tasks(research_plan, sample_query)
    executor.display_execution_results(results)

if __name__ == "__main__":
    main()