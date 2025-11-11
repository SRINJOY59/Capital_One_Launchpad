import os
import sys
from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(project_root)

from Tools.translation_tool import MultiLanguageTranslator

load_dotenv()

class SynthesizerAgent:
    def __init__(self):
        self.translator = MultiLanguageTranslator()
        self.agent = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            tools=[self.translate_text],
            instructions="""
You are a synthesis and summarization expert for agricultural AI. Your job is to take multiple responses from different agents (RAG, model, or tool outputs) and refactor them into a single, clear, actionable, and well-structured answer for the user.

INSTRUCTIONS:
- If any response is in a language other than English, use the Translator tool to give response .
- Carefully read and analyze each response in the provided list.
- Identify key points, insights, and recommendations from each response.
- Remove redundancy, resolve contradictions, and synthesize information into a coherent summary.
- Present the final result in a logical, readable format with clear sections, bullet points, and actionable advice.
- Use a professional, helpful, and concise tone.
- Do not mention tool calling or internal implementation details.
- The output should be suitable for direct presentation to the user.
If query is Hello or How are you? then you should also greet the user

"""
        )

    def translate_text(self, text: str, source_lang: str = 'auto', target_lang: str = 'en') -> str:
        result = self.translator.translate_robust(text, source_lang, target_lang)
        if result['status'] == 'success':
            return result['translated_text']
        else:
            return f"Translation failed: {result.get('error', 'Unknown error')}"

    def synthesize(self, responses: list[str]) -> str:
        prompt = (
            "Given the following responses from multiple agents, synthesize and refactor them into a single, clear, actionable, and well-structured answer for the user.\n\n"
            "Responses:\n"
        )
        for i, resp in enumerate(responses, 1):
            prompt += f"Response {i}:\n{resp}\n\n"
        prompt += "Provide the final synthesized answer below:\n"
        return self.agent.run(prompt).content

if __name__ == "__main__":
    responses = [
        "The weather forecast for Nashik indicates moderate rainfall and temperatures suitable for Kharif crops.",
        "Recommended crops for Nashik in Kharif season are rice, soybean, and maize due to local soil and climate conditions.",
        "Market prices for rice and soybean are favorable, and risk levels are moderate for these crops.",
        "Ekhon market e khub bhalo daam cholche."
    ]
    synthesizer = SynthesizerAgent()
    final_answer = synthesizer.synthesize(responses)
    print(final_answer)
