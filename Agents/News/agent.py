import os
import sys
from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
from agno.tools.tavily import TavilyTools
from agno.tools.baidusearch import BaiduSearchTools
from agno.tools.duckduckgo import DuckDuckGoTools
from pydantic import BaseModel



load_dotenv()

class NewsArticle(BaseModel):
    Content : str
    Links : list[str]

class NewsAgent:
    def __init__(self):
        self.agent = Agent(
            model=Gemini(id="gemini-2.0-flash"),  
            tools=[
                TavilyTools(),
                BaiduSearchTools(),
                DuckDuckGoTools()
            ],
            instructions="""
You are an agricultural news extraction agent. For any location or topic, extract and summarize 5 recent and relevant news articles about agriculture.
For each article, provide:
- Detailed information or search snippet about the article
- Valid URLs to the full articles

TOOL USAGE:
- Use TavilyTools, BaiduSearchTools, and DuckDuckGoTools to find the most recent and relevant agricultural news.
- If full article content cannot be extracted, provide the most recent search snippet and valid reference link for each news article.
- The reference links from TavilyTools are very important, always include them if available.
- If a tool only gives search results, use other tools to fetch and summarize the full article content if possible.
- Only include links that are valid and point to the actual article. If a link is not valid, do not show it.

OUTPUT FORMAT:
- Provide a detailed, compiled news summary with reference links.
- Do not mention tool calling in your response.
"""
        )

    def get_agri_news(self, location_or_topic: str):
        prompt = (
            f"Find recent and relevant news articles about agriculture for '{location_or_topic}'. "
            "If full article content cannot be extracted, provide the most recent search snippets and valid reference links for each news article. "
            "Only include links that are valid and point to the actual article."
        )
        result = self.agent.run(prompt).content
        return result
if __name__ == "__main__":
    news_agent = NewsAgent()
    location = "what is the current condition of Rice production in Haryana"
    news = news_agent.get_agri_news(location)
    print(news)