import os
import hashlib
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()

class Abstractor:
    def __init__(self, model_id="gemini-2.0-flash"):
        self.agent = Agent(
            model=Gemini(id=model_id),
            markdown=True,
            instructions='''You are an intelligent document analysis assistant that specializes in extracting and synthesizing information from retrieved documents. Your task is to analyze the provided content and generate accurate, relevant responses based on the given context.

            Here are your key responsibilities:

            1. **Content Analysis**: Carefully analyze all provided documents and context information.

            2. **Information Extraction**: Extract the most relevant and important information that directly addresses the user's query.

            3. **Response Generation**: 
               - Provide clear, concise, and accurate responses
               - Use only the information available in the provided context
               - Maintain factual accuracy and avoid hallucinations
               - Structure your response logically and coherently

            4. **Context Adherence**: 
               - Stay within the bounds of the provided documents
               - If information is not available in the context, clearly state this limitation
               - Do not make assumptions beyond what's provided

            5. **Quality Standards**:
               - Ensure responses are helpful and directly address the query
               - Use appropriate formatting for readability
               - Cite or reference specific parts of the documents when relevant
               - Maintain professional and informative tone

            6. **Handling Insufficient Information**: 
               If the provided context doesn't contain enough information to answer the query, respond with: 
               "Based on the provided documents, I don't have sufficient information to fully answer your query. Please provide additional context or try a more specific question."
            '''
        )
    
    def abstract(self, content, query=None):
        if query:
            prompt = f"Based on the following retrieved documents, please answer the query: '{query}'\n\nRetrieved Documents:\n{content}"
        else:
            prompt = f"Please analyze and summarize the key information from the following retrieved documents:\n\nDocuments:\n{content}"
        
        response = self.agent.run(prompt)
        return response.content
    
    def query_documents(self, query, documents):
        """Query specific documents with a user question"""
        prompt = f"Query: {query}\n\nRelevant Documents:\n{documents}\n\nPlease provide a comprehensive answer based on the above documents."
        response = self.agent.run(prompt)
        return response.content
    
    def summarize_documents(self, documents):
        """Summarize multiple documents"""
        prompt = f"Please provide a comprehensive summary of the following documents:\n\n{documents}"
        response = self.agent.run(prompt)
        return response.content

if __name__ == "__main__":
    abstractor = Abstractor()
    
    # Example 1: General content analysis
    content = """
    Document 1: Machine Learning Fundamentals
    Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.
    It includes supervised learning, unsupervised learning, and reinforcement learning approaches.
    
    Document 2: Neural Networks
    Neural networks are computing systems inspired by biological neural networks. They consist of nodes (neurons)
    connected by edges (synapses) that can learn complex patterns in data.
    """
    
    print("=== General Content Analysis ===")
    response = abstractor.abstract(content)
    print(response)
    
    print("\n=== Query-based Analysis ===")
    query = "What are the main types of machine learning?"
    response_with_query = abstractor.abstract(content, query)
    print(response_with_query)
    
    print("\n=== Document Querying ===")
    query_response = abstractor.query_documents(
        "Explain neural networks", 
        content
    )
    print(query_response)
    
    print("\n=== Document Summarization ===")
    summary = abstractor.summarize_documents(content)
    print(summary)