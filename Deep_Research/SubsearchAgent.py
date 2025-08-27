import os
import sys
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.arxiv import ArxivTools
from agno.tools.wikipedia import WikipediaTools
from agno.tools.tavily import TavilyTools
from Tools.fetchWeatherForecast import get_google_weather_forecast
from Tools.getCropRecommendation import get_crop_recommendation
from Tools.pest_prediction import detect_pests
from Tools.risk_management import get_agricultural_risk_metrics
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

@dataclass
class SearchResult:
    agent_name: str
    query: str
    content: str
    success: bool
    execution_time: float
    sources_found: int = 0
    error_message: Optional[str] = None

class EnhancedSubsearchAgent:
    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.agents = self._initialize_agents()
        
    def _initialize_agents(self) -> Dict[str, Agent]:
        common_tools = [
            get_google_weather_forecast, 
            get_crop_recommendation, 
            detect_pests, 
            get_agricultural_risk_metrics
        ]
        
        agents = {
            "arxiv": Agent(
                name="ArXiv Research Specialist",
                role="Academic research and scientific papers",
                model=Gemini(id="gemini-2.0-flash"),
                tools=[ArxivTools()] + common_tools,
                instructions="""
                You are the ArXiv Research Specialist focusing on academic agricultural research.
                - Search and analyze recent academic papers
                - Provide comprehensive literature reviews
                - Focus on peer-reviewed research and citations
                - Extract key findings and methodologies
                - Identify research gaps and trends
                Complete your analysis independently with detailed academic insights.
                """
            ),
            
            "wikipedia": Agent(
                name="Knowledge Base Specialist", 
                role="Foundational knowledge and context",
                model=Gemini(id="gemini-2.0-flash"),
                tools=[WikipediaTools()] + common_tools,
                instructions="""
                You are the Knowledge Base Specialist providing foundational agricultural knowledge.
                - Provide comprehensive background information
                - Explain fundamental concepts and definitions
                - Give historical context and established facts
                - Cross-reference information for accuracy
                - Build knowledge foundation for complex topics
                Complete your research independently with authoritative sources.
                """
            ),
            
            "tavily": Agent(
                name="Current Intelligence Specialist",
                role="Real-time information and market intelligence", 
                model=Gemini(id="gemini-2.0-flash"),
                tools=[TavilyTools()] + common_tools,
                instructions="""
                You are the Current Intelligence Specialist for real-time agricultural information.
                - Search current web developments and news
                - Provide market intelligence and trends
                - Find recent policy changes and regulations
                - Gather real-time agricultural data and statistics
                - Monitor emerging technologies and practices
                Complete your analysis independently with current, actionable insights.
                """
            )
        }
        
        return agents

    def _execute_single_search(self, agent_name: str, query: str, context: Optional[str] = None) -> SearchResult:
        start_time = datetime.now()
        agent = self.agents[agent_name]
        
        try:
            research_query = self._prepare_query(query, context, agent_name)
            response = agent.run(research_query)
            
            content = response.content if hasattr(response, 'content') else str(response)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            sources_count = self._count_sources(content)
            
            return SearchResult(
                agent_name=agent_name,
                query=query,
                content=content,
                success=True,
                execution_time=execution_time,
                sources_found=sources_count
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Search failed for {agent_name}: {str(e)}")
            
            return SearchResult(
                agent_name=agent_name,
                query=query,
                content="",
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )

    def _prepare_query(self, query: str, context: Optional[str], agent_name: str) -> str:
        base_query = f"Agricultural Research Query: {query}\n\n"
        
        if context:
            base_query += f"Context: {context}\n\n"
        
        agent_specific_instructions = {
            "arxiv": "Focus on academic papers, research methodologies, and peer-reviewed findings.",
            "wikipedia": "Provide foundational knowledge, definitions, and established facts.",
            "tavily": "Find current developments, market trends, and real-time data."
        }
        
        base_query += f"Instructions for {agent_name}:\n"
        base_query += agent_specific_instructions.get(agent_name, "Provide comprehensive analysis.")
        base_query += "\n\nProvide detailed analysis with sources, key findings, and actionable insights."
        
        return base_query

    def _count_sources(self, content: str) -> int:
        source_indicators = ['http', 'doi:', 'source:', 'reference:', 'cited', 'published']
        return sum(1 for indicator in source_indicators if indicator.lower() in content.lower())

    def subsearch_single_query(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a single query across all available agents and return combined results.
        This method is required by the workflow.py file.
        """
        start_time = datetime.now()
        results = []
        
        # Execute search across all agents for this single query
        for agent_name in self.agents.keys():
            result = self._execute_single_search(agent_name, query, context)
            results.append(result)
            
            if result.success:
                logger.info(f"Single query search - {agent_name}: Found {result.sources_found} sources")
            else:
                logger.warning(f"Single query search - {agent_name}: Search failed - {result.error_message}")
        
        total_time = (datetime.now() - start_time).total_seconds()
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        # Combine all successful results into a single content block
        combined_content = self._combine_single_query_results(successful_results, query)
        
        # Create sources list from successful results
        sources = []
        for result in successful_results:
            # Extract potential source information from content
            source_info = {
                "agent": result.agent_name,
                "query": result.query,
                "sources_found": result.sources_found,
                "execution_time": result.execution_time,
                "content_preview": result.content[:200] + "..." if len(result.content) > 200 else result.content
            }
            sources.append(source_info)
        
        return {
            "success": len(successful_results) > 0,
            "query": query,
            "combined_content": combined_content,
            "sources": sources,
            "sources_found": len(sources),
            "successful_searches": len(successful_results),
            "failed_searches": len(failed_results),
            "total_sources_found": sum(r.sources_found for r in successful_results),
            "execution_time": total_time,
            "agents_used": [r.agent_name for r in successful_results],
            "errors": [{"agent": r.agent_name, "error": r.error_message} for r in failed_results] if failed_results else []
        }

    def _combine_single_query_results(self, results: List[SearchResult], original_query: str) -> str:
        """Combine results from multiple agents for a single query"""
        if not results:
            return f"No successful results found for query: {original_query}"
        
        combined = f"# Research Results for: {original_query}\n\n"
        combined += f"*Compiled from {len(results)} successful agent searches*\n\n"
        
        # Group results by agent for better organization
        for i, result in enumerate(results, 1):
            combined += f"## {result.agent_name} Analysis\n"
            combined += f"**Sources Found:** {result.sources_found} | **Search Time:** {result.execution_time:.2f}s\n\n"
            combined += f"{result.content}\n\n"
            
            if i < len(results):
                combined += "---\n\n"
        
        # Add summary section
        total_sources = sum(r.sources_found for r in results)
        combined += f"\n## Search Summary\n"
        combined += f"- **Total Sources Found:** {total_sources}\n"
        combined += f"- **Agents Consulted:** {', '.join([r.agent_name for r in results])}\n"
        combined += f"- **Query Processed:** {original_query}\n"
        
        return combined

    def search_parallel(self, queries: List[str], context: Optional[str] = None) -> List[SearchResult]:
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for query in queries:
                for agent_name in self.agents.keys():
                    future = executor.submit(self._execute_single_search, agent_name, query, context)
                    futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=120)
                    all_results.append(result)
                    logger.info(f"Completed search: {result.agent_name} - {result.query[:50]}...")
                except Exception as e:
                    logger.error(f"Future execution failed: {str(e)}")
        
        return all_results

    def search_sequential(self, queries: List[str], context: Optional[str] = None) -> List[SearchResult]:
        all_results = []
        
        for query in queries:
            logger.info(f"Processing query: {query}")
            
            for agent_name in self.agents.keys():
                result = self._execute_single_search(agent_name, query, context)
                all_results.append(result)
                
                if result.success:
                    logger.info(f"Success {agent_name}: Found {result.sources_found} sources")
                else:
                    logger.warning(f"Failed {agent_name}: Search failed")
        
        return all_results

    def search_optimized(self, queries: List[str], context: Optional[str] = None, 
                        parallel: bool = True) -> Dict[str, Any]:
        start_time = datetime.now()
        
        if parallel and len(queries) > 1:
            results = self.search_parallel(queries, context)
            approach = "parallel_execution"
        else:
            results = self.search_sequential(queries, context)
            approach = "sequential_execution"
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        combined_content = self._combine_results(successful_results)
        
        return {
            "success": len(successful_results) > 0,
            "approach": approach,
            "total_execution_time": total_time,
            "queries_processed": len(queries),
            "successful_searches": len(successful_results),
            "failed_searches": len(failed_results),
            "total_sources_found": sum(r.sources_found for r in successful_results),
            "combined_content": combined_content,
            "detailed_results": successful_results,
            "errors": [{"agent": r.agent_name, "query": r.query, "error": r.error_message} 
                      for r in failed_results],
            "performance_metrics": self._calculate_performance_metrics(results)
        }

    def _combine_results(self, results: List[SearchResult]) -> str:
        if not results:
            return "No successful search results found."
        
        combined = f"# Comprehensive Agricultural Research Results\n"
        combined += f"*Generated from {len(results)} successful searches*\n\n"
        
        by_agent = {}
        for result in results:
            if result.agent_name not in by_agent:
                by_agent[result.agent_name] = []
            by_agent[result.agent_name].append(result)
        
        for agent_name, agent_results in by_agent.items():
            combined += f"## {agent_name.title()} Research Findings\n\n"
            
            for result in agent_results:
                combined += f"### Query: {result.query}\n"
                combined += f"**Sources Found:** {result.sources_found} | **Time:** {result.execution_time:.2f}s\n\n"
                combined += f"{result.content}\n\n"
                combined += "---\n\n"
        
        return combined

    def _calculate_performance_metrics(self, results: List[SearchResult]) -> Dict[str, float]:
        if not results:
            return {}
        
        successful = [r for r in results if r.success]
        
        return {
            "success_rate": len(successful) / len(results) * 100,
            "average_execution_time": sum(r.execution_time for r in results) / len(results),
            "average_sources_per_search": sum(r.sources_found for r in successful) / len(successful) if successful else 0,
            "fastest_search": min(r.execution_time for r in results),
            "slowest_search": max(r.execution_time for r in results)
        }

    def execute_parallel_research(self, research_queries: List[str], context: Optional[str] = None) -> Dict[str, Any]:
        """Execute parallel research across multiple queries and return comprehensive results"""
        return self.search_optimized(research_queries, context, parallel=True)

    def search(self, queries: List[str], context: Optional[str] = None, 
              parallel: bool = True) -> Dict[str, Any]:
        return self.search_optimized(queries, context, parallel)
    
if __name__ == "__main__":
    agent = EnhancedSubsearchAgent()

    query = "Soil Health Improvement in Wheat Farming"
    context = "Focus on sustainability, fertilizer management, and crop yield optimization."

    result = agent.subsearch_single_query(query, context)

    print("\n🔎 Subsearch Agent Results\n")
    if result["success"]:
        print(result["combined_content"])
        print("\n📌 Sources Summary:")
        for src in result["sources"]:
            print(f"- Agent: {src['agent']} | Sources Found: {src['sources_found']} | Time: {src['execution_time']:.2f}s")
            print(f"  Preview: {src['content_preview']}\n")
    else:
        print("❌ No successful results.")
        if result["errors"]:
            for err in result["errors"]:
                print(f"Agent {err['agent']} failed with error: {err['error']}")

