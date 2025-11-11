from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv

load_dotenv()

class QueryRewriterAgent:
    def __init__(self):
        self.agent = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            instructions="""
You are an expert query rewriting assistant for agricultural AI workflows. Your job is to improve user queries when the initial response is not satisfactory. 
INSTRUCTIONS:
- Carefully read the original query and the previous result.
- Consider the feedback from the answer grader.
- Rewrite the query to be clearer, more specific, and more likely to produce a high-quality answer.
- Remove ambiguity, add missing context, and focus the query on actionable information.
- Output only the rewritten query, suitable for re-processing by the workflow.
- Do not mention tool calling or internal implementation details.
"""
        )

    def rewrite(self, query: str, previous_result: str, feedback: str = "") -> str:
        prompt = (
            f"Original Query: {query}\n"
            f"Previous Result: {previous_result}\n"
            f"Grader Feedback: {feedback}\n"
            "Rewrite the query to improve the response quality."
        )
        return self.agent.run(prompt).content

if __name__ == "__main__":
    agent = QueryRewriterAgent()
    original_query = "What are the best crops for Kharif season in Nashik?"
    previous_result = "The best crops are rice and soybean."
    feedback = "The answer lacks details about soil, rainfall, and market prices."
    rewritten_query = agent.rewrite(original_query, previous_result, feedback)
    print(rewritten_query)
