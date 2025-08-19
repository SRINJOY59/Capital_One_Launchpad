from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class AnswerGradingResponse(BaseModel):
    feedback: str
    decision: bool

class AnswerGraderAgent:
    def __init__(self):
        self.agent = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            instructions="""
You are a helpful answer grading assistant. Your job is to quickly check if an answer is generally correct and relevant to the question.

INSTRUCTIONS:
- Read the question and answer.
- If the answer is mostly correct and relevant, give positive feedback and set decision=True.
- If the answer is mostly wrong or off-topic, give brief feedback and set decision=False.
- Be lenient and supportive. Do not over-penalize minor mistakes.
- If answer says that model doesn't have sufficient information to answer this question comprehensively, then obviously decision will be false
- Output should be an AnswerGradingResponse object with 'feedback' and 'decision'.
- Greetings type of query and answer can be given to you so answer accordingly
""",
            response_model=AnswerGradingResponse,
        )

    def grade(self, question: str, answer: str) -> AnswerGradingResponse:
        prompt = f"Question: {question}\nAnswer: {answer}\nIs the answer generally correct and relevant? Give brief feedback and decision."
        return self.agent.run(prompt).content

if __name__ == "__main__":
    grader = AnswerGraderAgent()
    question = "Explain the process of photosynthesis."
    answer = "Photosynthesis is the process by which plants make their food using sunlight, water, and carbon dioxide."
    result = grader.grade(question, answer)
    print(result)

