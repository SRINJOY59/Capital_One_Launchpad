import os
import sys
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.google_maps import GoogleMapTools
from agno.tools.tavily import TavilyTools
from agno.tools.googlesearch import GoogleSearchTools
from agno.team import Team
from dotenv import load_dotenv

load_dotenv()

# Setup paths for custom tools
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(project_root)

from Tools.translation_tool import MultiLanguageTranslator


class PersonalizedAssistant:
    def __init__(self, user_location, preferred_language, crops, total_land_area,
                 season=None, farming_type=None, irrigation=None,
                 budget=None, experience=None):
        """
        Initialize a Personalized Agricultural Assistant.

        Parameters:
        - user_location (str): The farmer’s location.
        - preferred_language (str): Language for output.
        - crops (list[str]): Crops grown by the user.
        - total_land_area (int): Land area in acres.
        - season (str, optional): Current farming season.
        - farming_type (str, optional): e.g., traditional / organic / mixed.
        - irrigation (str, optional): e.g., rainfed / canal / borewell.
        - budget (str, optional): e.g., low / medium / high.
        - experience (str, optional): e.g., beginner / experienced.
        """
        self.user_location = user_location
        self.preferred_language = preferred_language
        self.crops = crops
        self.total_land_area = total_land_area
        self.season = season
        self.farming_type = farming_type
        self.irrigation = irrigation
        self.budget = budget
        self.experience = experience

        # Custom tools
        self.translator = MultiLanguageTranslator()

        # Agent 1: Web search + agricultural practices
        self.web_search_agent = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            tools=[
                GoogleMapTools(),
                TavilyTools(),
                GoogleSearchTools(),
            ],
            instructions="""
                You are a personalized agricultural advisor and web search agent.
                Your expertise lies in gathering the latest agricultural insights and practical farming recommendations.
                Your responsibilities include:
                1. Recommending region-specific best practices for crop production.
                2. Sharing localized pest control and fertilizer updates.
                3. Identifying sustainable farming practices relevant to the user.
                4. Providing user-friendly summaries of the latest agricultural news and research.
            """
        )

        # Agent 2: Market & Policy insights
        self.market_agent = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            tools=[GoogleSearchTools(), TavilyTools()],
            instructions="""
                You are a market and agricultural policy advisor.
                Your responsibilities include:
                1. Providing updates on credit and loan policies, government subsidies, and crop insurance schemes.
                2. Highlighting market price trends for the user’s crops, including demand-supply dynamics.
                3. Offering guidance on government procurement rates and market opportunities.
                4. Presenting insights in a farmer-friendly way, avoiding excessive technical or financial jargon.
            """
        )

        # Agent 3: Translator / multilingual communication
        self.multi_lingual_agent = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            tools=[self.translator.translate_robust],
            instructions=f"""
                You are a multilingual communication specialist.
                Your goal is to adapt agricultural and market insights into the user’s preferred language ({self.preferred_language}).
                Ensure the response is natural, clear, and empathetic,
                so the user feels supported and confident in their farming decisions.
            """
        )

        # Team agent to coordinate all agents
        self.team_agent = Team(
            mode="coordinate",
            model=Gemini(id="gemini-2.0-flash"),
            members=[self.web_search_agent, self.market_agent, self.multi_lingual_agent],
            instructions=f"""
                You are a collaborative agricultural advisory team.
                Work together to deliver actionable farming guidance in the user’s preferred language.
                Tailor recommendations to the crops ({', '.join(self.crops)}) 
                and the land size ({self.total_land_area} acres).
                Ensure to cover:
                - Best agricultural practices
                - Fertilizer and pest control updates
                - Eco-friendly and sustainable methods
                - Credit and loan schemes, subsidies, insurance
                - Market price trends and demand forecasts
                Ensure the tone is friendly, respectful, and personalized, 
                avoiding technical jargon unless necessary.
            """,
            share_member_interactions=True,
        )

    def build_prompt(self):
        """Constructs a dynamic user profile prompt for personalization."""
        profile = f"""
        User Profile:
        - Location: {self.user_location}
        - Preferred Language: {self.preferred_language}
        - Crops: {', '.join(self.crops)}
        - Total Land Area: {self.total_land_area} acres
        """

        if self.season: profile += f"- Current Season: {self.season}\n"
        if self.farming_type: profile += f"- Farming Type: {self.farming_type}\n"
        if self.irrigation: profile += f"- Irrigation Method: {self.irrigation}\n"
        if self.budget: profile += f"- Budget: {self.budget}\n"
        if self.experience: profile += f"- Experience Level: {self.experience}\n"

        return f"""{profile}

        Task:
        Write a proper comprehensive response for this
        Provide:
        - Current best agricultural practices tailored to the crops and land size in the specified region.
        - Fertilizer usage, pesticide updates, irrigation methods, and eco-friendly practices.
        - Updates on credit policies, subsidies, and insurance relevant to the farmer.
        - Market price trends, demand forecasts, and local procurement opportunities.
        Deliver the information in {self.preferred_language}, keeping the tone warm, user-friendly, and actionable.
        """

    def run(self):
        """Runs the team agent with the built prompt."""
        prompt = self.build_prompt()
        return self.team_agent.run(prompt).content


if __name__ == "__main__":
    assistant = PersonalizedAssistant(
        user_location="Bankura",
        preferred_language="Bengali",
        crops=["Rice", "Wheat", "Maize"],
        total_land_area=100,
        season="Kharif",
        farming_type="Organic",
        irrigation="Rainfed",
        budget="Medium",
        experience="Beginner"
    )
    answer = assistant.run()
    print(answer)
