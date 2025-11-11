import os
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from agno.tools.tavily import TavilyTools
from dotenv import load_dotenv

load_dotenv()

class IntrospectiveAgent:
    def __init__(self, model_id="gemini-2.0-flash"):
        # Main reasoning agent
        self.main_agent = Agent(
            name="MainAgent",
            model=Gemini(id=model_id),
            tools=[TavilyTools()],
            markdown=True,
            instructions="""You are a knowledgeable assistant that helps users find information and answer questions. 
            
            Your capabilities:
            1. Search for current information using Tavily search when needed
            2. Analyze and synthesize information from multiple sources
            3. Provide comprehensive and accurate answers
            4. Ask clarifying questions when the query is ambiguous
            
            Always strive to provide helpful, accurate, and well-sourced information."""
        )
        
        # Self-reflection agent
        self.reflection_agent = Agent(
            name="ReflectionAgent",
            model=Gemini(id=model_id),
            markdown=True,
            instructions="""You are a critical thinking assistant that evaluates responses for quality and accuracy.
            
            Your role:
            1. Analyze the given response for completeness and accuracy
            2. Identify any gaps, inconsistencies, or areas for improvement
            3. Suggest specific improvements or additional information needed
            4. Evaluate if the response fully addresses the original question
            5. Rate the response quality and suggest refinements
            
            Be constructive and specific in your feedback."""
        )
        
        # Synthesis agent that combines insights
        self.synthesis_agent = Agent(
            name="SynthesisAgent", 
            model=Gemini(id=model_id),
            markdown=True,
            instructions="""You are a synthesis expert that creates improved responses based on original answers and reflection feedback.
            
            Your task:
            1. Take the original question, initial response, and reflection feedback
            2. Create an improved, comprehensive response that addresses all concerns
            3. Incorporate additional information or corrections as needed
            4. Ensure the final response is complete, accurate, and well-structured
            5. Maintain clarity and readability
            
            Provide only the improved final response."""
        )
    
    def introspect_and_respond(self, question: str, max_iterations: int = 1) -> str:
        current_response = None
        
        for iteration in range(max_iterations):
            print(f"\n=== Iteration {iteration + 1} ===")
            
            if iteration == 0:
                print("Generating initial response...")
                current_response = self.main_agent.run(question).content
                print(f"Initial Response: {current_response}")
            else:
                reflection_prompt = f"""
                Original Question: {question}
                
                Current Response: {current_response}
                
                Please analyze this response and provide specific feedback on how to improve it.
                """
                
                print("Reflecting on current response...")
                reflection = self.reflection_agent.run(reflection_prompt).content
                print(f"Reflection: {reflection}")
                
                synthesis_prompt = f"""
                Original Question: {question}
                
                Previous Response: {current_response}
                
                Reflection Feedback: {reflection}
                
                Create an improved response that addresses the feedback and better answers the original question.
                """
                
                print("Synthesizing improved response...")
                current_response = self.synthesis_agent.run(synthesis_prompt).content
                print(f"Improved Response: {current_response}")
        
        return current_response
    
    def simple_chat(self, question: str) -> str:
        return self.main_agent.run(question).content
    
    def reflect_on_response(self, question: str, response: str) -> str:
        reflection_prompt = f"""
        Question: {question}
        Response: {response}
        
        Analyze this response and provide constructive feedback.
        """
        return self.reflection_agent.run(reflection_prompt).content

if __name__ == "__main__":
    agent = IntrospectiveAgent()
    
    question = "Who is Mukti and what is their significance?"
    
    print("=== Introspective Agent Response ===")
    introspective_response = agent.introspect_and_respond(question, max_iterations=2)
    
    print(f"\n=== Final Introspective Response ===")
    print(introspective_response)
    
    print(f"\n" + "="*50)
    
    simple_response = agent.simple_chat(question)
    print(simple_response)
    
    print(f"\n=== Reflection on Simple Response ===")
    reflection = agent.reflect_on_response(question, simple_response)
    print(reflection)