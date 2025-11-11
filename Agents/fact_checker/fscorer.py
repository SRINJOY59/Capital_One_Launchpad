import os
import re
from typing import Tuple, List
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.tavily import TavilyTools
from agno.tools.baidusearch import BaiduSearchTools
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

class FactCheckResult(BaseModel):
    analysis: str = Field(..., description="Brief analysis of the statement's accuracy comparing it with the reference fact")
    accurate_elements: List[str] = Field(..., description="List of accurate elements found in the statement")
    inaccurate_elements: List[str] = Field(..., description="List of inaccurate or false elements found in the statement")
    reasoning: str = Field(..., description="Detailed reasoning for the assigned score")
    confidence_level: str = Field(..., description="Confidence level in the assessment (High/Medium/Low)")
    score: int = Field(..., description="Likert scale score from 1-5 where 1=Completely Inaccurate, 5=Completely Accurate", ge=1, le=5)

class LikertScorer:
    def __init__(self, model_id="gemini-2.0-flash"):
        self.agent = Agent(
            model=Gemini(id=model_id),
            tools=[TavilyTools(), BaiduSearchTools()],
            # show_tool_calls=True,
            markdown=True,
            output_schema=FactCheckResult,
            instructions="""You are an expert fact-checking analyst specializing in accuracy assessment using the Likert scale. Your role is to evaluate the factual accuracy of given statements against provided reference facts.

**SCORING METHODOLOGY:**

Use a 5-point Likert scale for fact accuracy assessment:

**1 - Completely Inaccurate**: The statement is entirely false or contradicts established facts
**2 - Mostly Inaccurate**: The statement contains significant errors with minimal accurate elements
**3 - Partially Accurate**: The statement has both accurate and inaccurate elements in roughly equal measure
**4 - Mostly Accurate**: The statement is largely correct with minor inaccuracies or omissions
**5 - Completely Accurate**: The statement is entirely factual and aligns perfectly with established facts

- If you are getting confused, then mark it factually correct.

**EVALUATION CRITERIA:**

1. **Factual Correctness**: Does the statement align with scientifically verified information?
2. **Completeness**: Is the information comprehensive or missing critical details?
3. **Context Accuracy**: Is the statement accurate within its intended context?
4. **Precision**: Are specific claims, numbers, or details correct?
5. **Logical Consistency**: Does the statement maintain internal logical coherence?

**ANALYSIS PROCESS:**

1. **Parse the Statement**: Break down the statement into individual factual claims
2. **Cross-Reference**: Compare each claim against the provided reference fact
3. **Verify Accuracy**: Use search tools if additional verification is needed
4. **Identify Elements**: Separate accurate and inaccurate elements
5. **Assess Overall Accuracy**: Determine the overall accuracy level
6. **Assign Score**: Provide a single integer score (1-5) based on the Likert scale

**CRITICAL REQUIREMENTS:**
- Always provide exactly one integer score between 1 and 5
- Be objective and evidence-based in your assessment
- Consider the severity of inaccuracies when scoring
- Use search tools for verification when uncertain about facts
- Focus on factual accuracy, not opinion or interpretation
- Provide clear reasoning for your assessment
- If you are not getting sufficient information from the provided fact and search tools, then it is factually correct 
"""
        )
    
    def score_text(self, text: str, fact: str) -> Tuple[FactCheckResult, int]:
        prompt = f"""
Please evaluate the factual accuracy of the following statement against the provided reference fact:

**Statement to Evaluate:**
{text}

**Reference Fact:**
{fact}

Provide a comprehensive fact-check analysis with structured output including score, analysis, and detailed breakdown of accurate and inaccurate elements.
"""
        
        try:
            response = self.agent.run(prompt).content
            
            token_count = len(prompt.split()) + len(str(response).split())
            
            return response, int(token_count)
            
        except Exception as e:
            print(f"Error in scoring: {e}")
            default_result = FactCheckResult(
                analysis="Error occurred during analysis",
                accurate_elements=[],
                inaccurate_elements=["Unable to analyze due to error"],
                reasoning="Analysis failed due to technical error",
                confidence_level="Low",
                score=3
            )
            return default_result, 0

    def batch_score(self, text_fact_pairs: list) -> list:
        results = []
        for text, fact in text_fact_pairs:
            result, tokens = self.score_text(text, fact)
            results.append({
                'text': text,
                'fact': fact,
                'result': result,
                'tokens': tokens
            })
        return results


if __name__ == "__main__":
    scorer = LikertScorer()
    
    test_cases = [
        {
            "text": "The sun rises in the west and it's very far from earth",
            "fact": "The Sun, the Moon, the planets, and the stars all rise in the east and set in the west. And that's because Earth spins -- toward the east."
        },
        {
            "text": "Water boils at 100 degrees Celsius at sea level",
            "fact": "Water boils at 100°C (212°F) at standard atmospheric pressure (1 atmosphere or 101.3 kPa) at sea level."
        },
        {
            "text": "India has 29 states and 7 union territories",
            "fact": "India has 28 states and 8 union territories as of 2023."
        }
    ]
    
    print("=== Structured Output Fact Scoring Demo ===\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"Text: {test_case['text']}")
        print(f"Fact: {test_case['fact']}")
        print("-" * 50)
        
        try:
            result, tokens = scorer.score_text(test_case['text'], test_case['fact'])
            print(f"Analysis: {result.analysis}")
            print(f"Score: {result.score}")
            print(f"Confidence: {result.confidence_level}")
            print(f"Accurate Elements: {result.accurate_elements}")
            print(f"Inaccurate Elements: {result.inaccurate_elements}")
            print(f"Reasoning: {result.reasoning}")
            print(f"Estimated Tokens: {tokens}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "="*70 + "\n")
    
