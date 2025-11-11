import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'Tools'))

from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.groq import Groq
from dotenv import load_dotenv
from agno.tools.tavily import TavilyTools
from Tools.web_scrapper import scrape_agri_prices, scrape_policy_updates, scrape_links

load_dotenv()

class AgriculturalWebScrappingAgent:
    def __init__(self, model_id = "gemini-2.0-flash"):
        if model_id == "gemini-2.0-flash":
            model = Gemini(id=model_id)
        else:
            model = Groq(id=model_id)
        self.agent = Agent(
            model=model,
            tools=[scrape_agri_prices, scrape_policy_updates, scrape_links, TavilyTools()],
            # show_tool_calls=True,
            # markdown=True,
            # add_history_to_messages=True,
            # num_history_responses=5,
            instructions="""
You are an agricultural web data extraction and search specialist. Your job is to provide actionable, easy-to-understand insights by scraping and searching for agricultural market prices, policy updates, and relevant links from the web.

WEB DATA & SEARCH FRAMEWORK:
- Use TavilyTools for web search to find authoritative sources, latest news, and relevant URLs for agricultural commodities, prices, and policies.
- Use scrape_agri_prices to extract tabular price data from mandi, exchange, or government sites identified via web search.
- Use scrape_policy_updates to extract latest agricultural policy news and updates from web pages found via search.
- Use scrape_links to gather URLs for further research or verification, prioritizing those surfaced by search tools.
- Always combine direct scraping with web search to ensure up-to-date and comprehensive results.
- Summarize extracted and searched data with clear, concise explanations and highlight key findings.
- Provide supporting statistics and facts directly from the scraped and searched web content.

PROMPTING STRATEGY:
- Clarify the commodity, location, and data type to be scraped or searched.
- Use TavilyTools to find the most recent and relevant web sources before scraping.
- Request specific tables, news updates, or links as needed, and verify their freshness via search.
- Present scraped and searched data in a readable format with bullet points or tables.
- Highlight trends, anomalies, and actionable insights.
- Include relevant web sources and URLs for reference.

OUTPUT REQUIREMENTS:
- Executive summary of findings.
- Tabular or bullet-point presentation of scraped and searched data.
- Key trends and actionable insights.
- Relevant web sources and links.
- Use simple language for broad understanding.
"""
        )

    def scrape(self, query: str) -> str:
        return self.agent.run(query).content

if __name__ == "__main__":
    agent = AgriculturalWebScrappingAgent(model_id="llama-3.3-70b-versatile")
    prompt = "Scrape and summarize the latest wheat mandi prices in Punjab and provide key trends and relevant links."
    print(agent.scrape(prompt))
