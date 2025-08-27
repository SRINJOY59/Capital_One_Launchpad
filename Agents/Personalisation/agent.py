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

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(project_root)

from Tools.web_scrapper import WebScrapper
from Tools.translation_tool import MultiLanguageTranslator


class PersonalizedAssistant:
    def __init__(self, user_location, preferred_language, crops, total_land_area):
        self.user_location = user_location
        self.preferred_language = preferred_language
        self.crops = crops
        self.total_land_area = total_land_area

        self.scrapper = WebScrapper()
        self.translator = MultiLanguageTranslator()

        self.web_search_agent = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            tools=[
                GoogleMapTools(),
                TavilyTools(),
                GoogleSearchTools(),
                self.scrapper.extract_table,
                self.scrapper.extract_text,
                self.scrapper.extract_links,
            ],
            instructions=f"""
                You are a personalized agricultural advisor and web search agent.
                Your expertise lies in gathering the latest agricultural insights and practical farming recommendations.
                Your responsibilities include:
                1. Recommending region-specific best practices for crop production.
                2. Sharing localized pest control and fertilizer updates.
                3. Identifying sustainable farming practices relevant to the user.
                4. Providing user-friendly summaries of the latest agricultural news and research.
                Personalize your response to the user’s crops, land size, and geographic location.
            """
        )

        self.multi_lingual_agent = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            tools=[self.translator.translate_robust],
            instructions=f"""
                You are a multilingual communication specialist.
                Your goal is to adapt agricultural insights into the user’s preferred language ({self.preferred_language}).
                Ensure the response is natural, clear, and empathetic, so the user feels supported and confident in their farming decisions.
            """
        )

        self.team_agent = Team(
            mode="coordinate",
            model=Gemini(id="gemini-2.0-flash"),
            members=[self.web_search_agent, self.multi_lingual_agent],
            instructions=f"""
                You are a collaborative agricultural advisory team.
                Work together to deliver actionable farming guidance in the user’s preferred language.
                Tailor recommendations to the crops ({', '.join(self.crops)}) and the land size ({self.total_land_area} acres).
                Ensure the tone is friendly, respectful, and personalized, avoiding technical jargon unless necessary.
            """,
            share_member_interactions=True,
        )

    def build_prompt(self):
        return f"""
            User Profile:
            - Location: {self.user_location}
            - Preferred Language: {self.preferred_language}
            - Crops: {', '.join(self.crops)}
            - Total Land Area: {self.total_land_area} acres

            Task:
            Write response within 250 words, that should be very crisp and concise
            Provide current best agricultural practices tailored to the above crops and land size in the specified region.
            Highlight fertilizer usage, pesticide updates, irrigation methods, and eco-friendly practices.
            Include relevant agricultural news and local insights that can benefit the user.
            Translate and deliver the information in {self.preferred_language}, keeping the tone warm, user-friendly, and actionable.
        """

    def run(self):
        prompt = self.build_prompt()
        return self.team_agent.run(prompt).content


if __name__ == "__main__":
    assistant = PersonalizedAssistant(
        user_location="Bankura",
        preferred_language="Bengali",
        crops=["Rice", "Wheat", "Maize"],
        total_land_area=100,
    )
    answer = assistant.run()
    print(answer)
