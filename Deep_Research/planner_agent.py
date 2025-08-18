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
    agricultural_domain: str = Field(..., description="Agricultural domain")
    assigned_agent: str = Field(..., description="Assigned agricultural specialist")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")
    priority: int = Field(default=1, description="Task priority (1-5)")
    expected_duration: str = Field(default="medium", description="Expected duration")
    subsearch_queries: List[str] = Field(default_factory=list, description="Subsearch queries")

class ResearchPlan(BaseModel):
    plan_id: str = Field(..., description="Plan identifier")
    title: str = Field(..., description="Research title")
    objective: str = Field(..., description="Research objective")
    tasks: List[Task] = Field(..., description="List of tasks")
    agent_assignments: Dict[str, List[str]] = Field(..., description="Agent to tasks mapping")
    execution_order: List[str] = Field(..., description="Task execution order")
    estimated_total_time: str = Field(default="unknown", description="Total estimated time")

class AgriculturalPlanningAgent:
    def __init__(self):
        load_dotenv()
        
        self.available_agents = {
            "soil_scientist": "Soil health, nutrients, pH, fertilizers, soil testing, micronutrients",
            "crop_agronomist": "Crop selection, planting, growth monitoring, pest control, diseases",
            "field_researcher": "Field trials, experimental design, data collection, statistical analysis",
            "data_analyst": "Agricultural data analysis, statistics, yield modeling, predictions",
            "climate_specialist": "Weather patterns, climate adaptation, seasonal planning, forecasting",
            "sustainability_expert": "Organic farming, sustainable practices, certification, environmental impact",
            "market_analyst": "Agricultural economics, pricing, supply chain, market trends",
            "technology_specialist": "Agricultural technology, IoT, precision farming, automation"
        }
        
        self.planner = Agent(
            name="Agricultural Planning Agent",
            model=Gemini(id="gemini-2.0-flash"),
            response_model=ResearchPlan,
            instructions=self._get_planning_instructions(),
        )

    def _get_planning_instructions(self) -> str:
        return f"""
        You are an Advanced Agricultural Research Planning Agent. Create comprehensive research plans with:
        
        1. TASK BREAKDOWN: Decompose objectives into specific, actionable tasks
        2. DOMAIN CLASSIFICATION: Categorize by agricultural domain
        3. SPECIALIST ASSIGNMENT: Match tasks with best-suited specialists
        4. DEPENDENCY MAPPING: Identify task dependencies and execution order
        5. SUBSEARCH PLANNING: Generate specific queries for each task
        
        Available specialists: {list(self.available_agents.keys())}
        
        For each task, provide:
        - Clear, specific task description
        - Appropriate agricultural domain
        - Best-matched specialist
        - Priority level (1-5, 5=highest)
        - List of subsearch queries for research
        - Dependencies on other tasks
        
        Create logical execution order considering dependencies.
        """

    def _determine_domain_and_agent(self, task_description: str, task_name: str) -> tuple:
        desc_lower = task_description.lower() + " " + task_name.lower()
        
        domain_mappings = {
            ("soil", "soil_scientist"): ["soil", "nutrient", "fertilizer", "ph", "micronutrient", "fertility"],
            ("crop", "crop_agronomist"): ["crop", "plant", "seed", "harvest", "yield", "variety", "pest", "disease"],
            ("field", "field_researcher"): ["field", "trial", "experiment", "test", "research", "study"],
            ("data", "data_analyst"): ["data", "analysis", "model", "statistics", "prediction", "trend"],
            ("climate", "climate_specialist"): ["climate", "weather", "season", "temperature", "rainfall", "drought"],
            ("sustainability", "sustainability_expert"): ["organic", "sustainable", "environment", "certification"],
            ("market", "market_analyst"): ["market", "price", "economic", "supply", "demand", "trade"],
            ("technology", "technology_specialist"): ["technology", "iot", "precision", "automation", "sensor", "digital"]
        }
        
        for (domain, agent), keywords in domain_mappings.items():
            if any(keyword in desc_lower for keyword in keywords):
                return domain, agent
        
        return "general", "crop_agronomist"

    def _generate_subsearch_queries(self, task_description: str, domain: str) -> List[str]:
        base_query = task_description.lower()
        
        queries = [
            f"latest research {base_query}",
            f"{domain} agricultural {base_query}",
            f"best practices {base_query}",
            f"case studies {base_query}",
            f"technology solutions {base_query}"
        ]
        
        return queries[:3]

    def create_plan(self, title: str, objective: str) -> ResearchPlan:
        try:
            planning_query = f"""
            Create a comprehensive research plan for:
            Title: {title}
            Objective: {objective}
            
            Generate 6-8 specific tasks covering all aspects of this agricultural research.
            Each task should be focused, actionable, and contribute to the overall objective.
            """
            
            response = self.planner.run(planning_query)
            
            if hasattr(response, 'content') and isinstance(response.content, ResearchPlan):
                plan = response.content
            else:
                plan = self._create_fallback_plan(title, objective)
            
            for task in plan.tasks:
                domain, agent = self._determine_domain_and_agent(task.description, task.name)
                task.agricultural_domain = domain
                task.assigned_agent = agent
                task.subsearch_queries = self._generate_subsearch_queries(task.description, domain)
            
            plan.agent_assignments = self._create_agent_assignments(plan.tasks)
            plan.execution_order = [task.task_id for task in sorted(plan.tasks, key=lambda t: t.priority, reverse=True)]
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creating plan: {str(e)}")
            return self._create_fallback_plan(title, objective)

    def _create_fallback_plan(self, title: str, objective: str) -> ResearchPlan:
        base_tasks = [
            {"name": "Literature Review", "description": f"Comprehensive literature review for {objective}", "priority": 5},
            {"name": "Current State Analysis", "description": f"Analyze current state of {objective}", "priority": 4},
            {"name": "Technology Assessment", "description": f"Evaluate available technologies for {objective}", "priority": 3},
            {"name": "Implementation Strategy", "description": f"Develop implementation strategy for {objective}", "priority": 3},
            {"name": "Risk Assessment", "description": f"Identify and assess risks related to {objective}", "priority": 2},
            {"name": "Cost-Benefit Analysis", "description": f"Economic analysis of {objective}", "priority": 2}
        ]
        
        tasks = []
        for i, task_info in enumerate(base_tasks):
            domain, agent = self._determine_domain_and_agent(task_info["description"], task_info["name"])
            
            task = Task(
                task_id=f"T{i+1:02d}",
                name=task_info["name"],
                description=task_info["description"],
                agricultural_domain=domain,
                assigned_agent=agent,
                priority=task_info["priority"],
                subsearch_queries=self._generate_subsearch_queries(task_info["description"], domain),
                dependencies=[f"T{i:02d}"] if i > 0 else []
            )
            tasks.append(task)
        
        return ResearchPlan(
            plan_id=f"AP{datetime.now().strftime('%Y%m%d%H%M')}",
            title=title,
            objective=objective,
            tasks=tasks,
            agent_assignments={},
            execution_order=[task.task_id for task in tasks]
        )

    def _create_agent_assignments(self, tasks: List[Task]) -> Dict[str, List[str]]:
        assignments = {agent: [] for agent in self.available_agents.keys()}
        
        for task in tasks:
            if task.assigned_agent in assignments:
                assignments[task.assigned_agent].append(task.task_id)
        
        return {agent: task_list for agent, task_list in assignments.items() if task_list}

    def display_plan(self, plan: ResearchPlan):
        print(f"\n{'='*60}")
        print(f"AGRICULTURAL RESEARCH PLAN: {plan.title}")
        print(f"{'='*60}")
        print(f"Objective: {plan.objective}")
        print(f"Plan ID: {plan.plan_id}")
        
        print(f"\nTasks ({len(plan.tasks)}):")
        for task in plan.tasks:
            print(f"\n{task.task_id}: {task.name} [Priority: {task.priority}]")
            print(f"  Domain: {task.agricultural_domain}")
            print(f"  Assigned: {task.assigned_agent}")
            print(f"  Dependencies: {', '.join(task.dependencies) if task.dependencies else 'None'}")
            print(f"  Subsearch Queries: {len(task.subsearch_queries)}")
        
        print(f"\nExecution Order: {' -> '.join(plan.execution_order)}")
        
        print(f"\nAgent Assignments:")
        for agent, task_ids in plan.agent_assignments.items():
            print(f"  {agent}: {', '.join(task_ids)}")


if __name__ == "__main__":
    agent = AgriculturalPlanningAgent()
    plan = agent.create_plan(
        title="Soil Health Improvement in Wheat Farming",
        objective="Develop strategies to enhance soil fertility and crop yield"
    )
    agent.display_plan(plan)
            