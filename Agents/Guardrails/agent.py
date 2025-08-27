import os
import sys
from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional

load_dotenv()

class GuardrailsResponse(BaseModel):
    is_agriculture_related: bool
    is_greeting: bool
    response_message: Optional[str] = None
    confidence_score: float  # 0.0 to 1.0
    category: str  # "agriculture", "greeting", "general", "inappropriate"

class AgriculturalGuardrailsAgent:
    def __init__(self, model_id="gemini-2.0-flash"):
        self.agent = Agent(
            model=Gemini(id=model_id),
            markdown=True,
            response_model=GuardrailsResponse,
            instructions="""
You are an agricultural query guardrails agent. Your role is to:

1. IDENTIFY QUERY TYPE:
   - Determine if the query is agriculture-related
   - Detect if it's a greeting/casual conversation
   - Classify inappropriate or irrelevant content

2. AGRICULTURE RELEVANCE:
   Answer "yes" (true) if the query is related to:
   - Crop cultivation, farming practices, agriculture techniques
   - Plant diseases, pest control, fertilizers, seeds
   - Weather forecasting for farming, irrigation, soil management
   - Agricultural equipment, machinery, tools
   - Livestock, animal husbandry, dairy farming
   - Market prices for agricultural commodities
   - Agricultural policies, subsidies, insurance, loans
   - Food production, harvest, post-harvest processing
   - Organic farming, sustainable agriculture
   - Agricultural research, biotechnology
   - Farm management, agricultural economics

3. GREETING DETECTION:
   Identify greetings like: "hello", "hi", "good morning", "how are you", "thank you", etc.

4. RESPONSE GENERATION:
   - For greetings: Provide warm, agriculture-focused welcome message
   - For agriculture queries: Set is_agriculture_related=true, no response message needed
   - For non-agriculture queries: Politely redirect to agriculture topics

5. CONFIDENCE SCORING:
   Rate your confidence (0.0-1.0) in the classification

6. CATEGORIZATION:
   - "agriculture": Farming/agriculture related queries
   - "greeting": Casual greetings, pleasantries
   - "general": Non-agriculture but appropriate queries
   - "inappropriate": Harmful, offensive, or completely irrelevant content

RESPONSE FORMAT:
{
    "is_agriculture_related": boolean,
    "is_greeting": boolean, 
    "response_message": "string or null",
    "confidence_score": float,
    "category": "string"
}

EXAMPLES:

Query: "Hello, how are you?"
Response: {
    "is_agriculture_related": false,
    "is_greeting": true,
    "response_message": "Hello! I'm here to help you with all your agricultural needs. Whether you have questions about crop cultivation, pest management, weather forecasting, or market prices, I'm ready to assist. How can I help you with your farming today?",
    "confidence_score": 0.95,
    "category": "greeting"
}

Query: "What's the best fertilizer for tomatoes?"
Response: {
    "is_agriculture_related": true,
    "is_greeting": false,
    "response_message": null,
    "confidence_score": 0.98,
    "category": "agriculture"
}

Query: "Tell me about cryptocurrency"
Response: {
    "is_agriculture_related": false,
    "is_greeting": false,
    "response_message": "I specialize in agricultural assistance. I can help you with farming practices, crop management, weather forecasting, market prices, and other agriculture-related topics. Is there anything farming-related I can help you with?",
    "confidence_score": 0.90,
    "category": "general"
}
"""
        )

    def evaluate_query(self, query: str) -> GuardrailsResponse:
        try:
            result = self.agent.run(f"Evaluate this query: '{query}'")
            
            # Handle different response types
            if hasattr(result, 'content'):
                response = result.content
            else:
                response = result
                
            # Ensure we have a valid GuardrailsResponse
            if not isinstance(response, GuardrailsResponse):
                # Fallback response if parsing fails
                return GuardrailsResponse(
                    is_agriculture_related=False,
                    is_greeting=False,
                    response_message="I'm here to help with agricultural questions. How can I assist you with farming today?",
                    confidence_score=0.5,
                    category="general"
                )
                
            return response
            
        except Exception as e:
            print(f"Guardrails evaluation error: {str(e)}")
            # Fallback response
            return GuardrailsResponse(
                is_agriculture_related=False,
                is_greeting=False,
                response_message="I'm here to help with agricultural questions. How can I assist you with farming today?",
                confidence_score=0.3,
                category="general"
            )

    def is_agriculture_query(self, query: str) -> bool:
        result = self.evaluate_query(query)
        return result.is_agriculture_related

    def handle_greeting(self, query: str) -> Optional[str]:
        result = self.evaluate_query(query)
        if result.is_greeting:
            return result.response_message
        return None

    def get_response_for_non_agriculture(self, query: str) -> str:
        result = self.evaluate_query(query)
        if not result.is_agriculture_related and result.response_message:
            return result.response_message
        
        return "I specialize in agricultural assistance. I can help you with farming practices, crop management, weather forecasting, market prices, and other agriculture-related topics. Is there anything farming-related I can help you with?"

if __name__ == "__main__":
    agent = AgriculturalGuardrailsAgent()
    
    test_queries = [
        "Hello, how are you?",
        "Good morning!",
        "Thank you for your help",
        "What's the best fertilizer for wheat?",
        "How to control pests in rice fields?",
        "What's the weather forecast for farming?",
        "Tell me about cryptocurrency",
        "What's the capital of France?",
        "How to cook pasta?",
        "Market price of corn today",
        "Agricultural subsidies in India",
        "Best irrigation methods for cotton"
    ]
    
    print("Agricultural Guardrails Agent Testing")
    print("="*60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = agent.evaluate_query(query)
        print(f"Agriculture Related: {result.is_agriculture_related}")
        print(f"Is Greeting: {result.is_greeting}")
        print(f"Category: {result.category}")
        print(f"Confidence: {result.confidence_score:.2f}")
        if result.response_message:
            print(f"Response: {result.response_message}")
        print("-" * 40)
