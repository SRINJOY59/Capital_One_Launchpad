from agno.agent import Agent
from agno.models.google import Gemini
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class VerificationRequest(BaseModel):
    answer : bool
    reasoning : str

class Verifier:
    def __init__(self):
        self.agent = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            instructions=(
                "You are a verification agent. "
                "Given a question, a model-generated answer, and a ground truth answer, "
                "compare the model's answer with the ground truth. "
                "Respond with True if they match, otherwise False. "
                "Provide a brief justification."
            ),
            response_model=VerificationRequest
        )

    def verify(self, question: str, model_answer: str, ground_truth: str):
        prompt = (
            f"Question: {question}\n"
            f"Model Answer: {model_answer}\n"
            f"Ground Truth: {ground_truth}\n"
            "Does the model answer match the ground truth? Respond True or False and justify."
        )
        return self.agent.run(prompt).content

# Example usage:
verifier = Verifier()
result = verifier.verify("What is the capital of France?", "Paris", "Paris")
print(result)