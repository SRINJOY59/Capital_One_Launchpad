import os
from pydantic import BaseModel, Field
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from dotenv import load_dotenv

load_dotenv()

class GradeHallucinations(BaseModel):
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )    
    
class HallucinationGrader:
    def __init__(self, model_id="gemini-2.0-flash"):
        self.agent = Agent(
            model=Gemini(id=model_id),
            response_model=GradeHallucinations,
            instructions="""You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.

            Your task is to evaluate if the generated answer is factually supported by the provided documents.

            Instructions:
            1. Carefully read the set of facts/documents provided
            2. Compare the LLM generation against these facts
            3. Assess whether the generation is grounded in and supported by the retrieved facts
            4. Give a binary score 'yes' or 'no'

            Scoring criteria:
            - 'yes': The answer is grounded in and supported by the set of facts
            - 'no': The answer contains information not supported by the facts or contradicts them

            Be strict in your evaluation - only mark 'yes' if the generation is truly supported by the provided facts.
            """
        )
        
    def grade_hallucinations(self, documents, generation):
        prompt = f"Set of facts: \n\n {documents} \n\n LLM generation: {generation}\n\nIs this generation grounded in and supported by the set of facts? Respond with 'yes' or 'no'."
        response = self.agent.run(prompt).content
        return response.binary_score

if __name__ == "__main__":
    grader = HallucinationGrader()
    
    # Test case 1: Grounded answer
    documents1 = "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed."
    generation1 = "Machine learning allows computers to learn from data and make decisions automatically."
    
    grade1 = grader.grade_hallucinations(documents1, generation1)
    print(f"Documents: {documents1}")
    print(f"Generation: {generation1}")
    print(f"Grade: {grade1}")
    print()
    
    # Test case 2: Hallucinated answer
    documents2 = "Python is a programming language known for its simplicity and readability."
    generation2 = "Python was invented in 1995 by John Smith and is primarily used for web development only."
    
    grade2 = grader.grade_hallucinations(documents2, generation2)
    print(f"Documents: {documents2}")
    print(f"Generation: {generation2}")
    print(f"Grade: {grade2}")