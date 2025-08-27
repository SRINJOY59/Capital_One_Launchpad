import os
from typing import Literal
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradeResponse(BaseModel):
    grade: Literal["yes", "no"] = Field(..., description="Grade result - yes if relevant and accurate, no if not")

class GraderAgent:
    def __init__(self):
        load_dotenv()
        
        self.grader = Agent(
            name="Agricultural Response Grader",
            model=Gemini(id="gemini-2.0-flash"),
            response_model=GradeResponse,
            instructions=self._get_grader_instructions(),
        )

    def _get_grader_instructions(self) -> str:
        return """You are an agricultural domain expert grader. Your task is to evaluate if a response properly addresses an agricultural query.

Grade "yes" if the response:
- Directly addresses the agricultural query
- Contains relevant agricultural information
- Provides actionable farming advice
- Includes specific agricultural recommendations
- Demonstrates understanding of agricultural concepts

Grade "no" if the response:
- Does not address the agricultural query
- Contains irrelevant or off-topic information
- Lacks agricultural context or expertise
- Provides generic non-agricultural advice
- Is factually incorrect about agricultural practices

Evaluate only relevance and accuracy to agricultural domain. Return only "yes" or "no"."""

    def grade_response(self, user_query: str, response: str) -> str:
        try:
            grading_prompt = f"""
            User Query: {user_query}
            
            Response to Grade: {response}
            
            Grade this response for agricultural relevance and accuracy.
            """
            
            result = self.grader.run(grading_prompt)
            
            if hasattr(result, 'content') and hasattr(result.content, 'grade'):
                return result.content.grade
            else:
                return "no"
                
        except Exception as e:
            logger.error(f"Error grading response: {str(e)}")
            return "no"

def main():
    grader = GraderAgent()
    
    test_query = "What fertilizer should I use for tomatoes?"
    test_response = "For tomatoes, use balanced NPK fertilizer 10-10-10 at planting, then switch to high potassium fertilizer during fruiting stage."
    
    grade = grader.grade_response(test_query, test_response)
    print(f"Query: {test_query}")
    print(f"Response Grade: {grade}")

if __name__ == "__main__":
    main()