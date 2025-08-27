import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskExecutionResult(BaseModel):
    task_id: str = Field(..., description="Task identifier")
    task_description: str = Field(..., description="Task description")
    execution_status: str = Field(..., description="Execution status (success/failed)")
    result_content: str = Field(..., description="Generated content from task execution")

class ExecutionPlan(BaseModel):
    execution_id: str = Field(..., description="Execution identifier")
    total_tasks: int = Field(..., description="Total number of tasks")
    completed_tasks: int = Field(default=0, description="Number of completed tasks")
    task_results: List[TaskExecutionResult] = Field(default_factory=list, description="Task execution results")
    overall_status: str = Field(default="pending", description="Overall execution status")

class TaskExecutorAgent:
    def __init__(self):
        load_dotenv()
        
        self.executor = Agent(
            name="Agricultural Task Executor",
            model=Gemini(id="gemini-2.0-flash"),
            instructions=self._get_executor_instructions(),
        )

    def _get_executor_instructions(self) -> str:
        return """You are an Agricultural Task Executor Agent specializing in agricultural knowledge and recommendations.

Your capabilities include:
- Fertilizer recommendations based on crop type, soil conditions, and growth stage
- Weather forecast analysis and agricultural implications
- Market price trends and selling recommendations
- Crop recommendations based on region, season, and soil type
- Crop yield predictions and optimization strategies
- Pest identification and management solutions
- Disease detection and treatment recommendations
- Risk assessment and mitigation strategies
- Agricultural best practices and farming techniques

When executing tasks:
1. Provide comprehensive, research-based agricultural advice
2. Include specific recommendations with quantities, timings, and methods
3. Consider regional factors, climate conditions, and seasonal variations
4. Offer practical, actionable solutions that farmers can implement
5. Include safety considerations and environmental impact
6. Provide cost-effective alternatives when applicable
7. Use scientific principles while keeping explanations farmer-friendly

Generate detailed responses that directly address the agricultural query with expert knowledge and practical guidance."""

    def execute_tasks(self, research_plan, user_query: str = "") -> ExecutionPlan:
        execution_plan = ExecutionPlan(
            execution_id=f"EX{datetime.now().strftime('%Y%m%d%H%M%S')}",
            total_tasks=len(research_plan.tasks),
            overall_status="in_progress"
        )
        
        for idx, task in enumerate(research_plan.tasks, start=1):
            try:
                task.task_id = f"TASK_{idx:03d}"
                logger.info(f"Executing {task.task_id}: {task.description}")
                
                content = self._generate_task_content(task.description, user_query)
                
                result = TaskExecutionResult(
                    task_id=task.task_id,
                    task_description=task.description,
                    execution_status="success",
                    result_content=content
                )
                
                execution_plan.task_results.append(result)
                execution_plan.completed_tasks += 1
                
            except Exception as e:
                logger.error(f"Error executing {task.task_id}: {str(e)}")
                
                result = TaskExecutionResult(
                    task_id=task.task_id,
                    task_description=task.description,
                    execution_status="failed",
                    result_content=f"Task execution failed: {str(e)}"
                )
                
                execution_plan.task_results.append(result)
        
        execution_plan.overall_status = "completed" if execution_plan.completed_tasks == execution_plan.total_tasks else "partial"
        return execution_plan

    def _generate_task_content(self, task_description: str, user_query: str) -> str:
        try:
            synthesis_prompt = f"""
            Original User Query: {user_query}
            
            Specific Task to Execute: {task_description}
            
            Provide comprehensive agricultural guidance for this task. Include:
            - Detailed recommendations and best practices
            - Specific quantities, timings, and application methods where applicable
            - Regional and seasonal considerations
            - Cost-effective solutions and alternatives
            - Safety precautions and environmental considerations
            - Step-by-step implementation guidance
            
            Make your response practical and actionable for farmers.
            """
            
            response = self.executor.run(synthesis_prompt)
            
            if hasattr(response, 'content'):
                return str(response.content)
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            return f"Unable to generate content for task: {task_description}. Error: {str(e)}"

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
            print(f"Status: {result.execution_status}")
            print(f"\nGenerated Content:")
            print(f"{result.result_content}")

def main():
    from planner_agent import AgriculturalPlanningAgent
    
    planner = AgriculturalPlanningAgent()
    executor = TaskExecutorAgent()
    
    sample_query = "I want to grow tomatoes and need fertilizer recommendations and market prices"
    
    research_plan = planner.create_plan(sample_query)
    planner.display_plan(research_plan)
    
    results = executor.execute_tasks(research_plan, sample_query)
    executor.display_execution_results(results)

if __name__ == "__main__":
    main()