import os
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Task(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Task name")
    description: str = Field(..., description="Task description")
    tool_assignment: str = Field(..., description="Assigned tool for task execution")
    priority: int = Field(default=1, description="Task priority (1-5)")
    expected_duration: str = Field(default="medium", description="Expected duration")

class ToolAssignmentSchema(BaseModel):
    tools_list: List[str] = Field(..., description="List of tools to be used for the query")
    task_descriptions: List[str] = Field(..., description="Brief descriptions of tasks to be performed")
    execution_priority: List[int] = Field(..., description="Priority order for task execution (1-5, 5=highest)")

class ResearchPlan(BaseModel):
    plan_id: str = Field(..., description="Plan identifier")
    title: str = Field(..., description="Research title")
    objective: str = Field(..., description="Research objective")
    tasks: List[Task] = Field(..., description="List of tasks")
    tools_list: List[str] = Field(..., description="List of tools to be used")
    execution_order: List[str] = Field(..., description="Task execution order")
    estimated_total_time: str = Field(default="unknown", description="Total estimated time")

class AgriculturalPlanningAgent:
    def __init__(self):
        load_dotenv()
        
        self.available_tools = {
            "Fertilizer Recommendation Tool": "Provides soil-specific fertilizer recommendations based on crop type, soil conditions, and nutrient requirements",
            "Web Search Tool": "Searches internet for latest agricultural research, market trends, best practices, and general information",
            "Market Price Tool": "Retrieves current market prices for crops, commodities, seeds, and agricultural inputs across different regions",
            "Weather Forecast Tool": "Provides weather predictions, climate data, seasonal forecasts, and weather-based farming recommendations",
            "Crop Recommendation Tool": "Suggests optimal crops based on soil type, climate conditions, location, and farmer preferences",
            "Crop Yield Prediction Tool": "Predicts expected crop yields based on historical data, current conditions, and farming practices",
            "Pest Detection Tool": "Identifies pests and provides management strategies using image recognition and symptom analysis",
            "Translation Tool": "Translates agricultural content between different languages for global accessibility",
            "Crop Disease Tool": "Diagnoses crop diseases from symptoms and provides treatment recommendations",
            "Google Maps Location Tool": "Provides geographical information, farm location mapping, and proximity to markets/resources",
            "Web Scrapper Tool": "Extracts specific agricultural data from websites, research papers, and online databases",
            "Risk Management Tool": "Assesses agricultural risks including weather, market, pest, and financial risks with mitigation strategies",
            "Agricultural News Tool": "Fetches latest agricultural news, policy updates, and industry developments",
            "Farm Credit Policy Tool": "Provides information on agricultural loans, subsidies, government schemes, and financial assistance"
        }
        
        self.planner = Agent(
            name="Agricultural Task Planning Agent",
            model=Gemini(id="gemini-2.0-flash"),
            response_model=ToolAssignmentSchema,
            instructions=self._get_planning_instructions(),
        )

    def _get_planning_instructions(self) -> str:
        tool_descriptions = "\n".join([f"- {tool}: {desc}" for tool, desc in self.available_tools.items()])
        
        return f"""You are an Agricultural Task Planning Agent that assigns appropriate tools to tasks based on user queries.

Available Tools:
{tool_descriptions}

Task Assignment Examples:

Query: "I need to know what fertilizer to use for my tomato crop"
Task: "Get fertilizer recommendations for tomato crop"
Tool: "Fertilizer Recommendation Tool"

Query: "What are the current prices of wheat in Punjab"
Task: "Check current wheat market prices in Punjab"
Tool: "Market Price Tool"

Query: "Will it rain next week for farming"
Task: "Get weather forecast for farming activities"
Tool: "Weather Forecast Tool"

Query: "What crops should I grow in sandy soil"
Task: "Get crop recommendations for sandy soil conditions"
Tool: "Crop Recommendation Tool"

Query: "How much yield can I expect from my rice field"
Task: "Predict rice crop yield estimation"
Tool: "Crop Yield Prediction Tool"

Query: "There are insects on my plants, what are they"
Task: "Identify pests on crop plants"
Tool: "Pest Detection Tool"

Query: "My crops have brown spots, what disease is this"
Task: "Diagnose crop disease from symptoms"
Tool: "Crop Disease Tool"

Query: "Find nearest agricultural market to my farm"
Task: "Locate nearby agricultural markets"
Tool: "Google Maps Location Tool"

Query: "What are the latest farming techniques for organic farming"
Task: "Research latest organic farming techniques"
Tool: "Web Search Tool"

Query: "Get data from agricultural research websites"
Task: "Extract agricultural research data"
Tool: "Web Scrapper Tool"

Query: "What are the risks of growing cotton this season"
Task: "Assess cotton farming risks this season"
Tool: "Risk Management Tool"

Query: "Latest news about agricultural policies"
Task: "Get latest agricultural policy news"
Tool: "Agricultural News Tool"

Query: "Information about farm loans and subsidies"
Task: "Find farm credit and subsidy information"
Tool: "Farm Credit Policy Tool"

For each user query, analyze and return:
- tools_list: List of specific tools needed (e.g., ["Fertilizer Recommendation Tool", "Market Price Tool"])
- task_descriptions: Brief descriptions of what each tool should accomplish
- execution_priority: Priority levels for each task (1-5, where 5 is highest priority)

Output only the tools list, task descriptions, and priorities that directly address the user's query."""

    def create_plan(self, user_query: str) -> ResearchPlan:
        try:
            planning_query = f"""
            User Query: {user_query}
            
            Create a comprehensive task plan that addresses this agricultural query.
            Break down the objective into specific, actionable tasks.
            Assign the most appropriate tool for each task.
            Ensure tasks flow logically and cover all aspects of the query.
            """
            
            response = self.planner.run(planning_query)
            
            if hasattr(response, 'content') and isinstance(response.content, ToolAssignmentSchema):
                schema_result = response.content
                plan = self._create_plan_from_schema(user_query, schema_result)
            else:
                plan = self._create_fallback_plan(user_query)
            
            plan.execution_order = [task.task_id for task in sorted(plan.tasks, key=lambda t: t.priority, reverse=True)]
            plan.tools_list = [task.tool_assignment for task in plan.tasks]
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creating plan: {str(e)}")
            return self._create_fallback_plan(user_query)

    def _create_plan_from_schema(self, user_query: str, schema: ToolAssignmentSchema) -> ResearchPlan:
        tasks = []
        for i, (tool, description, priority) in enumerate(zip(schema.tools_list, schema.task_descriptions, schema.execution_priority)):
            task = Task(
                task_id=f"T{i+1:02d}",
                name=f"Execute {tool}",
                description=description,
                tool_assignment=tool,
                priority=priority
            )
            tasks.append(task)
        
        return ResearchPlan(
            plan_id=f"AP{datetime.now().strftime('%Y%m%d%H%M')}",
            title=f"Agricultural Plan for: {user_query[:50]}...",
            objective=user_query,
            tasks=tasks,
            tools_list=schema.tools_list,
            execution_order=[task.task_id for task in sorted(tasks, key=lambda t: t.priority, reverse=True)]
        )

    def _create_fallback_plan(self, user_query: str) -> ResearchPlan:
        query_lower = user_query.lower()
        
        fallback_tasks = []
        
        if any(word in query_lower for word in ['fertilizer', 'nutrient', 'soil health']):
            fallback_tasks.append({"name": "Get Fertilizer Recommendations", "tool": "Fertilizer Recommendation Tool", "priority": 5})
        
        if any(word in query_lower for word in ['price', 'cost', 'market', 'sell']):
            fallback_tasks.append({"name": "Check Market Prices", "tool": "Market Price Tool", "priority": 4})
        
        if any(word in query_lower for word in ['weather', 'rain', 'climate', 'forecast']):
            fallback_tasks.append({"name": "Get Weather Information", "tool": "Weather Forecast Tool", "priority": 4})
        
        if any(word in query_lower for word in ['crop recommendation', 'what to grow', 'best crop']):
            fallback_tasks.append({"name": "Get Crop Recommendations", "tool": "Crop Recommendation Tool", "priority": 5})
        
        if not fallback_tasks:
            fallback_tasks = [
                {"name": "Research Query Information", "tool": "Web Search Tool", "priority": 4},
                {"name": "Get Market Context", "tool": "Market Price Tool", "priority": 3},
                {"name": "Check Weather Conditions", "tool": "Weather Forecast Tool", "priority": 3}
            ]
        
        tasks = []
        for i, task_info in enumerate(fallback_tasks):
            task = Task(
                task_id=f"T{i+1:02d}",
                name=task_info["name"],
                description=f"Execute {task_info['name'].lower()} for: {user_query}",
                tool_assignment=task_info["tool"],
                priority=task_info["priority"]
            )
            tasks.append(task)
        
        return ResearchPlan(
            plan_id=f"AP{datetime.now().strftime('%Y%m%d%H%M')}",
            title=f"Agricultural Plan for: {user_query[:50]}...",
            objective=user_query,
            tasks=tasks,
            tools_list=[task.tool_assignment for task in tasks],
            execution_order=[task.task_id for task in tasks]
        )

    def display_plan(self, plan: ResearchPlan):
        print(f"\n{'='*60}")
        print(f"AGRICULTURAL TASK PLAN")
        print(f"{'='*60}")
        print(f"Objective: {plan.objective}")
        print(f"Plan ID: {plan.plan_id}")
        
        print(f"\nTasks ({len(plan.tasks)}):")
        for task in plan.tasks:
            print(f"\n{task.task_id}: {task.name} [Priority: {task.priority}]")
            print(f"  Tool: {task.tool_assignment}")
            print(f"  Description: {task.description}")
        
        print(f"\nTools List: {plan.tools_list}")

def main():
    planner = AgriculturalPlanningAgent()
    
    sample_query = "I want to grow tomatoes in my farm, need fertilizer recommendations and market prices"
    plan = planner.create_plan(sample_query)
    planner.display_plan(plan)

if __name__ == "__main__":
    main()