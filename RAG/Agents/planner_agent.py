import os
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from dotenv import load_dotenv

load_dotenv()

class PlannerAgent:
    def __init__(self, model_id="gemini-2.0-flash"):
        self.agent = Agent(
            model=Gemini(id=model_id),
            markdown=True,
            instructions="""You are an AI planner expert in restructuring answers to completely resolve questions. When provided with an original question and a current answer that failed to resolve it, you must regenerate a better answer.

            Follow these steps to regenerate the answer:
            1. Analyze the original question for ambiguities or missing information
            2. Evaluate the current answer to identify why it failed
            3. Break down the problem into key components or sub-questions
            4. Regenerate the answer using a clear, chain-of-thought approach to ensure it resolves ambiguities
            5. Keep the response concise and to the point
            6. Validate the final answer

            Your goal is to provide a comprehensive, accurate answer that fully addresses the original question.
            """
        )
    
    def plan(self, question, current_answer=None):
        if current_answer:
            prompt = f"Here is the original question: \n\n{question}\n\nHere is the current answer that failed to resolve the question: \n\n{current_answer}\n\nRegenerate a better answer that completely resolves the original question."
        else:
            prompt = f"Here is the question that needs to be answered: \n\n{question}\n\nProvide a comprehensive answer."
        
        response = self.agent.run(prompt)
        return response.content
    
    def regenerate_answer(self, question, current_answer):
        return self.plan(question, current_answer)

if __name__ == "__main__":
    planner = PlannerAgent()
    
    # Test case 1: Regenerate answer
    question1 = "What is machine learning and how does it work?"
    current_answer1 = "Machine learning is AI."
    
    better_answer1 = planner.regenerate_answer(question1, current_answer1)
    print("=== Answer Regeneration ===")
    print(f"Question: {question1}")
    print(f"Current Answer: {current_answer1}")
    print(f"Better Answer: {better_answer1}")
    print()
    
    # Test case 2: Direct planning
    question2 = "Explain the differences between supervised and unsupervised learning."
    
    answer2 = planner.plan(question2)
    print("=== Direct Planning ===")
    print(f"Question: {question2}")
    print(f"Answer: {answer2}")