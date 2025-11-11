import os
from pydantic import BaseModel, Field
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from dotenv import load_dotenv

load_dotenv()

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )        

class Grader:
    def __init__(self, model_id="gemini-2.0-flash"):
        self.agent = Agent(
            model=Gemini(id=model_id),
            response_model=GradeDocuments,
            instructions="""You are a grader assessing relevance of a retrieved document to a user question.

            Your task is to evaluate if the retrieved document is relevant to the user's question.

            Instructions:
            1. Carefully read the retrieved document and the user question
            2. Assess if the document contains keywords or semantic meaning related to the user question
            3. Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question
            4. It does not need to be a stringent test - the goal is to filter out erroneous retrievals

            Scoring criteria:
            - 'yes': The document contains relevant information, keywords, or concepts related to the question
            - 'no': The document is completely unrelated or irrelevant to the question
            """
        )
        
    def grade_documents(self, question, document):
        prompt = f"Retrieved document: \n\n {document} \n\n User question: {question}\n\nIs this document relevant to the question? Respond with 'yes' or 'no'."
        response = self.agent.run(prompt).content
        return response.binary_score

if __name__ == "__main__":
    grader = Grader()
    
    # Test case 1: Relevant document
    question1 = "What is machine learning?"
    document1 = "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention."
    
    grade1 = grader.grade_documents(question1, document1)
    print(f"Question: {question1}")
    print(f"Document: {document1}")
    print(f"Grade: {grade1}")
    print()
    
    # Test case 2: Irrelevant document
    question2 = "What is machine learning?"
    document2 = "Pizza is a dish of Italian origin consisting of a usually round, flat base of leavened wheat-based dough topped with tomatoes, cheese, and often various other ingredients."
    
    grade2 = grader.grade_documents(question2, document2)
    print(f"Question: {question2}")
    print(f"Document: {document2}")
    print(f"Grade: {grade2}")