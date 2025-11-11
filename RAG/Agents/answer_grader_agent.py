import os
from pydantic import BaseModel, Field
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from dotenv import load_dotenv

load_dotenv()

class GradeAnswer(BaseModel):
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )    
    
class AnswerGrader:
    def __init__(self, model_id="gemini-2.0-flash"):
        self.agent = Agent(
            model=Gemini(id=model_id),
            response_model=GradeAnswer,
            instructions="""You are a grader assessing whether an answer addresses / resolves a question. 

            Your task is to evaluate if the provided answer adequately addresses the given question.

            Instructions:
            1. Carefully read the user question and the LLM-generated answer
            2. Assess whether the answer directly addresses and resolves the question
            3. Give a binary score: 'yes' if the answer resolves the question, 'no' if it doesn't
            4. Be strict in your evaluation - the answer must actually address the core of the question

            Scoring criteria:
            - 'yes': The answer directly addresses the question and provides relevant information
            - 'no': The answer is off-topic, incomplete, or doesn't address the question
            """
        )
    
    def grade_answer(self, question, generation):
        prompt = f"User question: {question}\n\nLLM generation: {generation}\n\nDoes this answer address and resolve the question? Respond with 'yes' or 'no'."
        response = self.agent.run(prompt).content
        return response.binary_score

if __name__ == "__main__":
    grader = AnswerGrader()
    
    question1 = "What is machine learning?"
    answer1 = "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data without being explicitly programmed."
    
    grade1 = grader.grade_answer(question1, answer1)
    print(f"Question: {question1}")
    print(f"Answer: {answer1}")
    print(f"Grade: {grade1}")
    print()
    
    # Test case 2: Poor answer
    question2 = "What is machine learning?"
    answer2 = "The weather is nice today and I like pizza."
    
    grade2 = grader.grade_answer(question2, answer2)
    print(f"Question: {question2}")
    print(f"Answer: {answer2}")
    print(f"Grade: {grade2}")