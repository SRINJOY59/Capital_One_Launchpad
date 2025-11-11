from pprint import pprint
from .adaptive_rag_class import ADAPTIVE_RAG
from .stategraph import GraphState
from langgraph.graph import END, StateGraph, START
import os
    
class Workflow:
    def __init__(self, model, api_key, k, file_path, cache_dir=None):
        self.model = model
        self.api_key = api_key  
        self.k = k
        self.file_path = file_path
        
        if cache_dir:
            self.custom_cache_dir = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
        else:
            self.custom_cache_dir = None
        
        self.adaptive_rag = ADAPTIVE_RAG(
            model=model, 
            api_key=api_key, 
            k=k, 
            file_path=file_path,
            cache_dir=cache_dir
        )
        
        self.workflow = StateGraph(GraphState)
        
        self.workflow.add_node("retrieve", self.adaptive_rag.retrieve)
        self.workflow.add_node("grade_documents", self.adaptive_rag.grade_documents)
        self.workflow.add_node("simple_query_handler", self.adaptive_rag.simple_query_handler)
        self.workflow.add_node("moderate_query_handler", self.adaptive_rag.moderate_query_handler)
        self.workflow.add_node("complex_query_handler", self.adaptive_rag.complex_query_handler)
        self.workflow.add_node("transform_query", self.adaptive_rag.transform_query)
        self.workflow.add_node("web_search", self.adaptive_rag.web_search)
        self.workflow.add_node("introspective_agent_response", self.adaptive_rag.introspective_agent_response)
        
        self.workflow.add_edge(START, "retrieve")
        
        self.workflow.add_edge("retrieve", "grade_documents")
        
        self.workflow.add_conditional_edges(
            "grade_documents",
            self.adaptive_rag.adaptive_router,
            {
                "simple_path": "simple_query_handler",
                "moderate_path": "moderate_query_handler", 
                "complex_path": "complex_query_handler",
            },
        )
        
        self.workflow.add_edge("simple_query_handler", END)
        
        self.workflow.add_edge("complex_query_handler", END)
        
        self.workflow.add_conditional_edges(
            "moderate_query_handler",
            self.adaptive_rag.grade_generation_v_documents_and_question,
            {
                "not supported": "introspective_agent_response",
                "useful": END,
                "not useful": "transform_query",
            },
        )
        
        self.workflow.add_edge("transform_query", "retrieve")
        self.workflow.add_edge("introspective_agent_response", END)
        
        self.app = self.workflow.compile()

    def build_workflow(self):
        pass

    def run_workflow(self, inputs):
        workflow_path = []
        final_result = None
        
        for output in self.app.stream(inputs):
            for key, value in output.items():
                workflow_path.append(key)
                pprint(f"Node '{key}':")
                if key in ["simple_query_handler", "moderate_query_handler", "complex_query_handler", "introspective_agent_response"]:
                    final_result = value
                    if key == "complex_query_handler" and value.get("used_introspective_agent"):
                        print("✓ Introspective agent was used for quality improvement")
            pprint("\n---\n")

        print(f"Workflow path taken: {' -> '.join(workflow_path)}")
        
        if final_result is None:
            final_result = {}
            
        return {
            "generation": final_result.get("generation", "No generation available"),
            "extracted_info": final_result.get("extractions"),
            "documents": final_result.get("documents", []),
            "workflow_type": final_result.get("workflow_type", "moderate"),
            "workflow_path": workflow_path,
            "used_introspective_agent": final_result.get("used_introspective_agent", False),
            "initial_generation": final_result.get("initial_generation", None)
        }
