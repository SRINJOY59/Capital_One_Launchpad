import os
from typing import List, Dict, Any, Literal, TypedDict
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
import logging
from datetime import datetime
import concurrent.futures
import threading
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

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
    grade: str = Field(default="no", description="Grade from grader agent")
    iteration: int = Field(default=1, description="Iteration number")
    status: str = Field(default="success", description="Execution status")

class GradeResponse(BaseModel):
    grade: Literal["yes", "no"] = Field(..., description="Grade result - yes if relevant and accurate, no if not")

class WorkflowState(TypedDict):
    user_query: str
    research_plan: ResearchPlan
    agent_results: List[SubsearchAgentResult]
    final_results: List[SubsearchAgentResult]
    execution_id: str
    current_iteration: int
    max_iterations: int

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

    def process_query(self, query: str, previous_result: str = "", iteration: int = 1) -> str:
        try:
            if iteration > 1 and previous_result:
                enhanced_query = f"""
                Original Query: {query}
                Previous attempt result: {previous_result}
                
                The previous response was not satisfactory. Please improve by:
                1. Adding more specific agricultural details
                2. Including actionable recommendations
                3. Providing clearer explanations
                4. Adding practical implementation steps
                5. Ensuring agricultural accuracy and relevance
                
                Using your assigned tools ({', '.join(self.assigned_tools)}), provide comprehensive agricultural guidance.
                """
            else:
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

class GraderAgent:
    def __init__(self):
        load_dotenv()
        
        self.grader = Agent(
            name="Agricultural Response Grader",
            model=Gemini(id="gemini-2.0-flash"),
            response_model=GradeResponse,
            instructions=self._get_grader_instructions(),
        )

    def _get_grader_instructions(self) -> str:
        return """You are an agricultural domain expert grader. Your task is to evaluate if a response properly addresses an agricultural query.

Grade "yes" if the response:
- Directly addresses the agricultural query
- Contains relevant agricultural information
- Provides actionable farming advice
- Includes specific agricultural recommendations
- Demonstrates understanding of agricultural concepts

Grade "no" if the response:
- Does not address the agricultural query
- Contains irrelevant or off-topic information
- Lacks agricultural context or expertise
- Provides generic non-agricultural advice
- Is factually incorrect about agricultural practices

Evaluate only relevance and accuracy to agricultural domain. Return only "yes" or "no"."""

    def grade_response(self, user_query: str, response: str) -> str:
        try:
            grading_prompt = f"""
            User Query: {user_query}
            
            Response to Grade: {response}
            
            Grade this response for agricultural relevance and accuracy.
            """
            
            result = self.grader.run(grading_prompt)
            
            if hasattr(result, 'content') and hasattr(result.content, 'grade'):
                return result.content.grade
            else:
                return "no"
                
        except Exception as e:
            logger.error(f"Error grading response: {str(e)}")
            return "no"

class AgriculturalWorkflow:
    def __init__(self, max_iterations: int = 3):
        self.planner = AgriculturalPlanningAgent()
        self.grader = GraderAgent()
        self.subsearch_agents = {}
        self.max_iterations = max_iterations
        
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(WorkflowState)
        
        workflow.add_node("plan_tasks", self._plan_tasks_node)
        workflow.add_node("execute_agents", self._execute_agents_node)
        workflow.add_node("grade_responses", self._grade_responses_node)
        workflow.add_node("refine_responses", self._refine_responses_node)
        workflow.add_node("finalize_results", self._finalize_results_node)
        
        workflow.set_entry_point("plan_tasks")
        
        workflow.add_edge("plan_tasks", "execute_agents")
        workflow.add_edge("execute_agents", "grade_responses")
        
        workflow.add_conditional_edges(
            "grade_responses",
            self._should_continue_refinement,
            {
                "refine": "refine_responses",
                "finalize": "finalize_results"
            }
        )
        
        workflow.add_edge("refine_responses", "execute_agents")
        workflow.add_edge("finalize_results", END)
        
        return workflow.compile(checkpointer=MemorySaver())

    def _plan_tasks_node(self, state: WorkflowState) -> WorkflowState:
        research_plan = self.planner.create_plan(state["user_query"])
        state["research_plan"] = research_plan
        state["current_iteration"] = 1
        return state

    def _execute_single_agent_task(self, task: Task, user_query: str, agent_counter: int, current_iteration: int, previous_results: Dict[str, str]) -> SubsearchAgentResult:
        agent_name = f"Agent_{agent_counter:02d}_{task.tool_assignment.replace(' ', '_')}"
        assigned_tools = [task.tool_assignment]
        
        thread_id = threading.current_thread().ident
        logger.info(f"Thread {thread_id}: Executing {agent_name} - Iteration {current_iteration}")
        
        if agent_name not in self.subsearch_agents:
            self.subsearch_agents[agent_name] = SubsearchAgent(agent_name, assigned_tools)
        
        try:
            previous_result = previous_results.get(agent_name, "")
            result_content = self.subsearch_agents[agent_name].process_query(user_query, previous_result, current_iteration)
            
            agent_result = SubsearchAgentResult(
                agent_name=agent_name,
                assigned_tools=assigned_tools,
                query=task.description,
                result=result_content,
                iteration=current_iteration,
                status="success"
            )
            
        except Exception as e:
            agent_result = SubsearchAgentResult(
                agent_name=agent_name,
                assigned_tools=assigned_tools,
                query=task.description,
                result=f"Error: {str(e)}",
                iteration=current_iteration,
                status="failed"
            )
        
        logger.info(f"Thread {thread_id}: Completed {agent_name} - Iteration {current_iteration}")
        return agent_result

    def _execute_agents_node(self, state: WorkflowState) -> WorkflowState:
        research_plan = state["research_plan"]
        current_iteration = state["current_iteration"]
        
        previous_results = {}
        if state.get("agent_results"):
            for result in state["agent_results"]:
                previous_results[result.agent_name] = result.result
        
        agent_results = []
        max_workers = min(len(research_plan.tasks), 10)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self._execute_single_agent_task, task, state["user_query"], counter, current_iteration, previous_results): (task, counter)
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
                        iteration=current_iteration,
                        status="failed"
                    )
                    agent_results.append(error_result)
        
        agent_results.sort(key=lambda x: int(x.agent_name.split('_')[1]))
        state["agent_results"] = agent_results
        
        return state

    def _grade_responses_node(self, state: WorkflowState) -> WorkflowState:
        user_query = state["user_query"]
        agent_results = state["agent_results"]
        
        for result in agent_results:
            if result.status == "success":
                grade = self.grader.grade_response(user_query, result.result)
                result.grade = grade
            else:
                result.grade = "no"
        
        return state

    def _should_continue_refinement(self, state: WorkflowState) -> str:
        current_iteration = state["current_iteration"]
        max_iterations = state.get("max_iterations", self.max_iterations)
        
        if current_iteration >= max_iterations:
            return "finalize"
        
        agent_results = state["agent_results"]
        needs_refinement = any(result.grade == "no" and result.status == "success" for result in agent_results)
        
        if needs_refinement:
            return "refine"
        else:
            return "finalize"

    def _refine_responses_node(self, state: WorkflowState) -> WorkflowState:
        state["current_iteration"] += 1
        return state

    def _finalize_results_node(self, state: WorkflowState) -> WorkflowState:
        state["final_results"] = state["agent_results"]
        return state

    def execute_workflow(self, user_query: str) -> WorkflowState:
        initial_state = WorkflowState(
            user_query=user_query,
            execution_id=f"WF{datetime.now().strftime('%Y%m%d%H%M%S')}",
            current_iteration=1,
            max_iterations=self.max_iterations,
            agent_results=[],
            final_results=[]
        )
        
        config = {
            "configurable": {
                "thread_id": f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
        }
        
        final_state = self.workflow.invoke(initial_state, config)
        return final_state

    def execute_workflow_as_string(self, user_query: str) -> str:
        """Execute workflow and return results as a formatted string"""
        final_state = self.execute_workflow(user_query)
        return self.format_results_as_string(final_state)

    def format_results_as_string(self, final_state: WorkflowState) -> str:
        """Convert workflow results to a comprehensive string format"""
        result_string = f"""
AGRICULTURAL WORKFLOW RESULTS
{'='*80}
Execution ID: {final_state['execution_id']}
Original Query: {final_state['user_query']}
Total Iterations: {final_state['current_iteration']}
Total Agents: {len(final_state['final_results'])}
Status: Completed

RESEARCH PLAN SUMMARY
{'-'*40}
Plan ID: {final_state['research_plan'].plan_id}
Title: {final_state['research_plan'].title}
Objective: {final_state['research_plan'].objective}
Total Tasks: {len(final_state['research_plan'].tasks)}
Tools Used: {', '.join(final_state['research_plan'].tools_list)}

AGENT EXECUTION RESULTS
{'-'*40}
"""
        
        for i, result in enumerate(final_state["final_results"], 1):
            result_string += f"""
AGENT {i}: {result.agent_name}
Status: {result.status} | Grade: {result.grade} | Iteration: {result.iteration}
Assigned Tools: {', '.join(result.assigned_tools)}
Task Query: {result.query}

Response:
{result.result}

{'-'*60}
"""
        
        # Add summary of successful vs failed agents
        successful_agents = [r for r in final_state["final_results"] if r.status == "success"]
        high_grade_agents = [r for r in final_state["final_results"] if r.grade == "yes"]
        
        result_string += f"""
EXECUTION SUMMARY
{'-'*40}
Successful Agents: {len(successful_agents)}/{len(final_state['final_results'])}
High Grade Responses: {len(high_grade_agents)}/{len(final_state['final_results'])}
Total Iterations Used: {final_state['current_iteration']}
Max Iterations Allowed: {final_state['max_iterations']}

CONSOLIDATED RECOMMENDATIONS
{'-'*40}
"""
        
        # Consolidate all successful agent responses
        all_recommendations = []
        for result in successful_agents:
            if result.grade == "yes":
                all_recommendations.append(f"• {result.assigned_tools[0]}: {result.result[:200]}...")
        
        if all_recommendations:
            result_string += "\n".join(all_recommendations)
        else:
            result_string += "No high-grade recommendations available."
        
        result_string += f"\n\n{'='*80}\nWorkflow Execution Complete"
        
        return result_string

    def get_simple_answer(self, user_query: str) -> str:
        """Get a simple, consolidated answer from all agents"""
        final_state = self.execute_workflow(user_query)
        
        # Collect all successful responses
        successful_responses = []
        for result in final_state["final_results"]:
            if result.status == "success" and result.grade == "yes":
                successful_responses.append(result.result)
        
        if not successful_responses:
            return "Unable to generate satisfactory agricultural guidance. Please refine your query and try again."
        
        # Create a consolidated response
        consolidated = f"""
Agricultural Guidance for: {user_query}

Based on comprehensive analysis using multiple agricultural tools and expertise:

"""
        
        for i, response in enumerate(successful_responses, 1):
            consolidated += f"{i}. {response}\n\n"
        
        consolidated += """
This guidance is based on current agricultural best practices and expert recommendations. 
Please consult with local agricultural experts for region-specific advice.
"""
        
        return consolidated.strip()

    def get_executive_summary(self, user_query: str) -> str:
        """Get an executive summary of the workflow results"""
        final_state = self.execute_workflow(user_query)
        
        successful_count = len([r for r in final_state["final_results"] if r.status == "success"])
        high_grade_count = len([r for r in final_state["final_results"] if r.grade == "yes"])
        
        summary = f"""
EXECUTIVE SUMMARY - Agricultural Analysis

Query: {user_query}
Execution ID: {final_state['execution_id']}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Performance Metrics:
- Total Agents Deployed: {len(final_state['final_results'])}
- Successful Executions: {successful_count}
- High-Quality Responses: {high_grade_count}
- Iterations Completed: {final_state['current_iteration']}

Tools Utilized: {', '.join(final_state['research_plan'].tools_list)}

Key Findings:
"""
        
        for result in final_state["final_results"]:
            if result.status == "success" and result.grade == "yes":
                summary += f"- {result.assigned_tools[0]}: Analysis completed successfully\n"
        
        if high_grade_count == 0:
            summary += "- Recommendation: Query may need refinement for better results\n"
        
        summary += f"\nOverall Assessment: {'Successful' if high_grade_count > 0 else 'Needs Improvement'}"
        
        return summary

def main():
    workflow = AgriculturalWorkflow(max_iterations=3)
    
    sample_query = "I want to grow tomatoes in my farm, need fertilizer recommendations and market prices"
    
    print("Testing different output formats:")
    print("\n" + "="*80)
    print("1. FULL STRING FORMAT")
    print("="*80)
    
    start_time = datetime.now()
    full_result = workflow.execute_workflow_as_string(sample_query)
    end_time = datetime.now()
    
    print(full_result)
    print(f"\nExecution time: {(end_time - start_time).total_seconds():.2f} seconds")
    
    print("\n" + "="*80)
    print("2. SIMPLE ANSWER FORMAT")
    print("="*80)
    
    simple_answer = workflow.get_simple_answer(sample_query)
    print(simple_answer)
    
    print("\n" + "="*80)
    print("3. EXECUTIVE SUMMARY FORMAT")
    print("="*80)
    
    executive_summary = workflow.get_executive_summary(sample_query)
    print(executive_summary)

if __name__ == "__main__":
    main()