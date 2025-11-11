import os
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from dotenv import load_dotenv

load_dotenv()

class QuestionRewriter:
    def __init__(self, model_id="gemini-2.0-flash"):
        self.rewriter_agent = Agent(
            model=Gemini(id=model_id),
            markdown=True,
            instructions="""You are a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.
            
            Your task:
            1. Analyze the input question for its core semantic intent
            2. Reformulate the question to be more specific and retrieval-friendly
            3. Ensure the rewritten question maintains the original meaning
            4. Make it more likely to match relevant documents in a vector database
            
            Provide only the improved question, nothing else."""
        )
        
        self.hyde_agent = Agent(
            model=Gemini(id=model_id),
            markdown=True,
            instructions="""You are a HyDE (Hypothetical Document Embeddings) generator. Your task is to generate a hypothetical answer or document that would contain the information needed to answer the given question.
            
            Your task:
            1. Analyze the input question
            2. Generate a hypothetical, detailed answer or document passage that would contain the information to answer this question
            3. Write as if you are creating content that would appear in a relevant document
            4. Be specific and detailed in your hypothetical response
            5. Focus on the type of information that would be found in documents that could answer this question
            
            Generate only the hypothetical document content, nothing else."""
        )
    
    def re_write_question(self, question):
        prompt = f"Here is the initial question: \n\n {question} \n\nFormulate an improved question."
        response = self.rewriter_agent.run(prompt)
        return response.content
    
    def hyde_transformation(self, question):
        prompt = f"Generate a hypothetical document passage that would contain the information to answer this question: \n\n{question}"
        hypothetical_doc = self.hyde_agent.run(prompt)
        
        # Combine original question with hypothetical document for enhanced retrieval
        final_response = f"{question}\n\nHypothetical relevant content:\n{hypothetical_doc.content}"
        return final_response
    
    def get_hypothetical_answer(self, question):
        prompt = f"Generate a hypothetical document passage that would contain the information to answer this question: \n\n{question}"
        response = self.hyde_agent.run(prompt)
        return response.content

if __name__ == "__main__":
    rewriter = QuestionRewriter()
    
    original_question = "What is ML?"
    print("=== Question Rewriting ===")
    print(f"Original Question: {original_question}")
    
    better_question = rewriter.re_write_question(original_question)
    print(f"Rewritten Question: {better_question}")
    print()
    
    print("=== HyDE Transformation ===")
    hyde_result = rewriter.hyde_transformation(original_question)
    print(f"HyDE Enhanced Query:\n{hyde_result}")
    print()
    
    print("=== Hypothetical Answer Only ===")
    hyp_answer = rewriter.get_hypothetical_answer(original_question)
    print(f"Hypothetical Answer: {hyp_answer}")
    
    complex_question = "How does gradient descent work in neural networks?"
    print(f"\n=== Complex Question Test ===")
    print(f"Original: {complex_question}")
    
    rewritten = rewriter.re_write_question(complex_question)
    print(f"Rewritten: {rewritten}")
    
    hyde_enhanced = rewriter.hyde_transformation(complex_question)
    print(f"HyDE Enhanced:\n{hyde_enhanced}")