import os
import sys
from dataclasses import dataclass
from typing import Optional, List
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.google_maps import GoogleMapTools
from agno.tools.tavily import TavilyTools
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.reasoning import ReasoningTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.team import Team

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(project_root)

from Tools.commodities_price_forcasting import predict_future_days
from Tools.fetchMarketPrice import fetch_market_price
from Tools.fetchWeatherForecast import get_google_weather_forecast
from Tools.getCropYield import crop_yield_inference
from Tools.risk_management import get_agricultural_risk_metrics
from Tools.web_scrapper import WebScrapper


@dataclass
class AgentConfig:
    name: str
    instructions: str
    tools: list
    model_id: str = "gemini-2.0-flash"


class AgentFactory:
    def __init__(self):
        self.web_scrapper = WebScrapper()

    def build_location_agent(self) -> Agent:
        cfg = AgentConfig(
            name="Location Agent",
            instructions=(
                "You operate as a geospatial intelligence function focused on agricultural risk surfaces. "
                "Given any place or boundary, triangulate precise coordinates, administrative hierarchies, and proximal assets relevant to agriculture, logistics, and relief operations. "
                "Synthesize outputs that are immediately actionable for government agencies, cooperatives, NGOs, and insurers to drive field deployment, last-mile advisories, and policy triggers."
            ),
            tools=[GoogleMapTools(), TavilyTools(), GoogleSearchTools(), DuckDuckGoTools()],
        )
        return Agent(
            name=cfg.name,
            model=Gemini(id=cfg.model_id),
            tools=cfg.tools,
            add_name_to_instructions=True,
            instructions=cfg.instructions,
        )

    def build_weather_agent(self) -> Agent:
        cfg = AgentConfig(
            name="Weather Agent",
            instructions=(
                "You function as a climate-risk forecaster. "
                "Ingest short- and medium-range forecasts and surface agri-critical signals: heavy rainfall, heat stress, cold snaps, wind gusts, humidity windows, and dry spells. "
                "Map weather to crop phenology and recommend timing for sowing, irrigation, pesticide scheduling, and harvest to minimize loss and optimize input use for administrators and FPOs."
            ),
            tools=[
                get_google_weather_forecast,
                self.web_scrapper.extract_text,
                self.web_scrapper.extract_links,
                self.web_scrapper.extract_table,
                TavilyTools(),
                DuckDuckGoTools(),
                GoogleSearchTools(),
            ],
        )
        return Agent(
            name=cfg.name,
            model=Gemini(id=cfg.model_id),
            tools=cfg.tools,
            add_name_to_instructions=True,
            instructions=cfg.instructions,
        )

    def build_market_agent(self) -> Agent:
        cfg = AgentConfig(
            name="Market & Risk Agent",
            instructions=(
                "You are the risk and market intelligence core. "
                "Fuse mandi prices, commodity futures, crop yield signals, and local constraints to quantify forward-looking exposure for farmers and administrators. "
                "Use provided tools to fetch prices, project near-term price trajectories, infer yield deltas, and compute risk metrics (weather, pest, market volatility, and input shocks). "
                "Deliver clear mitigation playbooks: procurement buffers, MSP escalation signals, targeted advisories, input credit rationalization, crop switching, and localized contingency actions."
            ),
            tools=[
                fetch_market_price,
                predict_future_days,
                crop_yield_inference,
                get_agricultural_risk_metrics,
                TavilyTools(),
            ],
        )
        return Agent(
            name=cfg.name,
            model=Gemini(id=cfg.model_id),
            tools=cfg.tools,
            add_name_to_instructions=True,
            instructions=cfg.instructions,
        )


class RiskMitigationOrchestrator:
    def __init__(self, mode: str = "collaborate", model_id: str = "gemini-2.0-flash"):
        self.factory = AgentFactory()
        self.location_agent = self.factory.build_location_agent()
        self.weather_agent = self.factory.build_weather_agent()
        self.market_agent = self.factory.build_market_agent()
        self.team = Team(
            name="Agriculture Risk Response Team",
            mode=mode,
            model=Gemini(id=model_id),
            members=[self.location_agent, self.weather_agent, self.market_agent],
            instructions=(
                "You are a cross-functional risk command center engineered to translate multi-sourced signals into mitigation measures. "
                "For every query, produce outputs tailored for government departments, district administrations, cooperatives, and agri-enterprises. "
                "Prioritize early warnings, quantified risk levels, resource allocation guidance, and operational SOPs suitable for rapid rollout. "
                "When uncertain, propose low-regret hedges and phased interventions."
            ),
            tools=[ReasoningTools()],
            enable_agentic_context=True,
            markdown=True,
        )

    def build_prompt(self, geography: str, objectives: Optional[List[str]] = None) -> str:
        obj = objectives or [
            "Quantify acute and chronic agricultural risks",
            "Recommend actionable mitigation for administrators and FPOs",
            "Optimize timing of interventions across the crop calendar",
            "Safeguard smallholders through market and climate hedges",
        ]
        line_items = "\n".join([f"- {o}" for o in obj])
        return (
            f"Context: You are advising on agricultural risk mitigation for {geography}.\n"
            f"Objectives:\n{line_items}\n"
            f"Deliverables:\n"
            f"- A concise risk brief with likelihood, impact, and time horizon.\n"
            f"- A playbook of immediate (0-7 days), near-term (8-30 days), and seasonal (1-3 months) actions.\n"
            f"- Specific guidance for government, cooperatives/FPOs, input suppliers, and farmers.\n"
            f"- Monitoring KPIs and data hooks to track effectiveness.\n"
            f"Constraints: Be precise, evidence-led, and implementation-ready."
        )

    def run(self, geography: str, objectives: Optional[List[str]] = None) -> str:
        prompt = self.build_prompt(geography, objectives)
        result = self.team.run(prompt)
        return getattr(result, "content", str(result))


if __name__ == "__main__":
    orchestrator = RiskMitigationOrchestrator()
    output = orchestrator.run("Medinipur")
    print(output)
