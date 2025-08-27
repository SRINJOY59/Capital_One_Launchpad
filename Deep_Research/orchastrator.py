import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
import logging
from datetime import datetime
import concurrent.futures
import threading

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

class SubsearchAgentResult(BaseModel):
    agent_name: str = Field(..., description="Name of the subsearch agent")
    assigned_tools: List[str] = Field(..., description="Tools assigned to this agent")
    query: str = Field(..., description="Query processed by this agent")
    result: str = Field(..., description="Result from the subsearch agent")
    status: str = Field(default="success", description="Execution status")

class MultiAgentResult(BaseModel):
    execution_id: str = Field(..., description="Execution identifier")
    original_query: str = Field(..., description="Original user query")
    total_agents: int = Field(..., description="Total number of subsearch agents")
    agent_results: List[SubsearchAgentResult] = Field(default_factory=list, description="Results from all subsearch agents")
    overall_status: str = Field(default="completed", description="Overall execution status")

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

For each user query, analyze and return:
- tools_list: List of specific tools needed
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

class SubsearchAgent:
    def __init__(self, agent_name: str, assigned_tools: List[str]):
        self.agent_name = agent_name
        self.assigned_tools = assigned_tools
        
        self.agent = Agent(
            name=agent_name,
            model=Gemini(id="gemini-2.0-flash"),
            instructions=self._get_agent_instructions(),
        )

    def _get_agent_instructions(self) -> str:
        tool_descriptions = []
        available_tools = {
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
        
        for tool in self.assigned_tools:
            if tool in available_tools:
                tool_descriptions.append(f"- {tool}: {available_tools[tool]}")
        
        tools_text = "\n".join(tool_descriptions)
        
        return f"""You are a specialized Agricultural Subsearch Agent: {self.agent_name}

Your assigned tools:
{tools_text}

Your role is to provide expert agricultural advice using your assigned tools. 
When responding to queries:
1. Use your assigned tools' capabilities to provide comprehensive answers
2. Give specific, actionable recommendations
3. Include practical implementation steps
4. Consider regional and seasonal factors
5. Provide cost-effective solutions
6. Include safety and environmental considerations
7. Make responses farmer-friendly and practical

Generate detailed, expert-level agricultural guidance based on your specialized tool set."""

    def process_query(self, query: str) -> str:
        try:
            enhanced_query = f"""
            Query: {query}
            
            Using your assigned tools ({', '.join(self.assigned_tools)}), provide comprehensive agricultural guidance.
            Include specific recommendations, implementation steps, and practical advice.
            """
            
            response = self.agent.run(enhanced_query)
            
            if hasattr(response, 'content'):
                return str(response.content)
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"Error in {self.agent_name}: {str(e)}")
            return f"Error processing query with {self.agent_name}: {str(e)}"

class MultiAgentOrchestrator:
    def __init__(self):
        self.planner = AgriculturalPlanningAgent()
        self.subsearch_agents = {}

    def _execute_agent_task(self, task: Task, user_query: str, agent_counter: int) -> SubsearchAgentResult:
        agent_name = f"Agent_{agent_counter:02d}_{task.tool_assignment.replace(' ', '_')}"
        assigned_tools = [task.tool_assignment]
        
        thread_id = threading.current_thread().ident
        logger.info(f"Thread {thread_id}: Executing {agent_name}")
        
        if agent_name not in self.subsearch_agents:
            self.subsearch_agents[agent_name] = SubsearchAgent(agent_name, assigned_tools)
        
        try:
            result_content = self.subsearch_agents[agent_name].process_query(user_query)
            
            agent_result = SubsearchAgentResult(
                agent_name=agent_name,
                assigned_tools=assigned_tools,
                query=task.description,
                result=result_content,
                status="success"
            )
            
        except Exception as e:
            agent_result = SubsearchAgentResult(
                agent_name=agent_name,
                assigned_tools=assigned_tools,
                query=task.description,
                result=f"Error: {str(e)}",
                status="failed"
            )
        
        logger.info(f"Thread {thread_id}: Completed {agent_name}")
        return agent_result

    def execute_query(self, user_query: str, max_workers: int = None) -> MultiAgentResult:
        research_plan = self.planner.create_plan(user_query)
        
        execution_id = f"MA{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        if max_workers is None:
            max_workers = min(len(research_plan.tasks), 10)
        
        agent_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self._execute_agent_task, task, user_query, counter): (task, counter)
                for counter, task in enumerate(research_plan.tasks, 1)
            }
            
            for future in concurrent.futures.as_completed(future_to_task):
                task, counter = future_to_task[future]
                try:
                    agent_result = future.result()
                    agent_results.append(agent_result)
                except Exception as e:
                    logger.error(f"Error in task execution: {str(e)}")
                    agent_name = f"Agent_{counter:02d}_{task.tool_assignment.replace(' ', '_')}"
                    error_result = SubsearchAgentResult(
                        agent_name=agent_name,
                        assigned_tools=[task.tool_assignment],
                        query=task.description,
                        result=f"Thread execution error: {str(e)}",
                        status="failed"
                    )
                    agent_results.append(error_result)
        
        agent_results.sort(key=lambda x: int(x.agent_name.split('_')[1]))
        
        return MultiAgentResult(
            execution_id=execution_id,
            original_query=user_query,
            total_agents=len(agent_results),
            agent_results=agent_results,
            overall_status="completed"
        )

    def display_results(self, results: MultiAgentResult):
        print(f"\n{'='*80}")
        print(f"MULTI-AGENT EXECUTION RESULTS")
        print(f"{'='*80}")
        print(f"Execution ID: {results.execution_id}")
        print(f"Original Query: {results.original_query}")
        print(f"Total Agents: {results.total_agents}")
        print(f"Status: {results.overall_status}")
        
        for i, agent_result in enumerate(results.agent_results, 1):
            print(f"\n{'-'*60}")
            print(f"AGENT {i}: {agent_result.agent_name}")
            print(f"Status: {agent_result.status}")
            print(f"Assigned Tools: {', '.join(agent_result.assigned_tools)}")
            print(f"Query: {agent_result.query}")
            print(f"\nResult:")
            print(f"{agent_result.result}")

def main():
    orchestrator = MultiAgentOrchestrator()
    
    sample_query = "I want to grow tomatoes in my farm, need fertilizer recommendations and market prices"
    
    start_time = datetime.now()
    results = orchestrator.execute_query(sample_query, max_workers=5)
    end_time = datetime.now()
    
    print(f"Execution completed in: {(end_time - start_time).total_seconds():.2f} seconds")
    orchestrator.display_results(results)

if __name__ == "__main__":
    main()