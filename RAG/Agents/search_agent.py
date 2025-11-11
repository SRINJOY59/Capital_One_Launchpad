import os
from agno.agent import Agent, RunResponse
from agno.tools.tavily import TavilyTools
from agno.models.google import Gemini
from dotenv import load_dotenv

load_dotenv()

class Search:
    def __init__(self, k=3, model_id="gemini-2.0-flash"):
        self.k = k
        self.agent = Agent(
            model=Gemini(id=model_id),
            tools=[TavilyTools()],
            show_tool_calls=True,
            markdown=True,
            instructions=f"""You are a web search assistant that helps users find current and relevant information.
            
            Your capabilities:
            1. Search for current information using Tavily web search
            2. Analyze and summarize search results
            3. Provide accurate, up-to-date information
            4. Return top {k} most relevant results when possible
            
            Always use the search tool to find the most current and accurate information for user queries."""
        )
    
    def web_search(self, question):
        prompt = f"Search for information about: {question}"
        response = self.agent.run(prompt)
        return response.content
    
    def search_and_format(self, question):
        prompt = f"Search for detailed information about: {question}. Provide a comprehensive summary of the findings."
        response = self.agent.run(prompt)
        return response.content

if __name__ == "__main__":
    search_agent = Search(k=5)
    
    test_queries = [
        "Recent breakthroughs in quantum computing",
        "Latest developments in artificial intelligence",
        "Current weather in New York City",
        "Recent news about electric vehicles",
        "Latest stock market trends"
    ]
    
    
    for query in test_queries:
        print(f"Searching for: {query}")
        print("=" * 50)        
        results = search_agent.web_search(query)
        print(results)
        
        print("\n" + "-" * 80 + "\n")