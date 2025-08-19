import os
import sys
from typing import Dict, Any, List, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import StateGraph, END, START

from RAG.workflow import Workflow
from RAG.parallel_rag_main import ParallelRAGSystem
from Agents.Router import RouterAgent
from Agents.Crop_Recommender.agent import CropRecommenderAgent
from Agents.Weather_forcast.agent import WeatherForecastAgent
from Agents.Location_Information.agent import LocationAgriAssistant
from Agents.News.agent import NewsAgent
from Agents.Credit_Policy_Market.agent import CreditPolicyMarketAgent
from Agents.answer_grader import AnswerGraderAgent
from Agents.synthesizer_agent import SynthesizerAgent
from Agents.Crop_Disease.agent import CropDiseaseAgent
from Agents.fact_checker.fscorer import LikertScorer
from Agents.Image_Analysis.agent import ImageAgent
from Agents.Market_Price.agent import MarketPriceAgent
from Agents.Multi_Lingual.agent import MultiLingualAgent
from Agents.Pest_prediction.agent import PestPredictionAgent
from Agents.Risk_Management.agent import AgriculturalRiskAnalysisAgent
from Agents.Web_Scrapping.agent import AgriculturalWebScrappingAgent
from Agents.Crop_Yield.agent import CropYieldAssistant
from Agents.Query_rewriter import QueryRewriterAgent
from Agents.Fertilizer_Recommender.agent import FertilizerRecommendationAgent
from utils.Internet_checker import InternetChecker
from utils.hf_model import HFModel

internet_checker = InternetChecker()
base_model_dir = "./models/Qwen1.5-Base"
adapter_dir = "./models/Qwen_1.5_Finetuned"
hf_model = None

if not internet_checker.is_connected():
    print("Offline mode detected. Using HF Model for inference.")
    hf_model = HFModel(base_model_dir, adapter_dir)

else: 
    print("Online mode detected")

crop_recommender_agent = CropRecommenderAgent()
weather_forecast_agent = WeatherForecastAgent()
location_agri_assistant = LocationAgriAssistant()
news_agent = NewsAgent()
crop_yield_assistant = CropYieldAssistant()
credit_policy_market_agent = CreditPolicyMarketAgent()
answer_grader_agent = AnswerGraderAgent()
synthesizer_agent = SynthesizerAgent()
crop_disease_agent = CropDiseaseAgent()
fact_checker_agent = LikertScorer()
image_analysis_agent = ImageAgent()
market_price_agent = MarketPriceAgent()
multi_language_translator_agent = MultiLingualAgent()
pest_prediction_agent = PestPredictionAgent()
risk_management_agent = AgriculturalRiskAnalysisAgent()
web_scraping_agent = AgriculturalWebScrappingAgent()
query_rewriter_agent = QueryRewriterAgent()
fertilizer_recommender_agent = FertilizerRecommendationAgent()

class MainWorkflowState(TypedDict):
    query: str
    image_path: str
    initial_mode: str
    current_mode: str
    rag_response: str
    router_result: Dict[str, Any]
    agent_responses: Dict[str, Any]
    fact_checked_responses: Dict[str, Any]
    synthesized_result: str
    grading: Any
    answer_grade: Any
    rewritten_query: str
    generation: str
    extractions: str
    documents: List[str]
    has_switched_mode: bool
    is_image_query: bool

def run_adaptive_rag(query: str) -> str:
    rag_system = ParallelRAGSystem(model="gemini-2.0-flash", k=3)
    result = rag_system.process_query(query)
    return result.get("synthesized_answer", "")

def run_router_agent(query: str, image_path: str = None) -> Dict[str, Any]:
    router = RouterAgent()
    if image_path:
        query_with_image = f"{query} [IMAGE_PROVIDED]"
        routing_decision = router.route(query_with_image)
    else:
        routing_decision = router.route(query)
    
    if hasattr(routing_decision, 'agents'):
        agents = routing_decision.agents
    elif isinstance(routing_decision, dict):
        agents = routing_decision.get('agents', [])
    else:
        agents = []
    return {"agents": agents, "routing_decision": routing_decision}

def call_agent(agent_name: str, query: str, image_path: str = None) -> Any:
    if agent_name == "CropRecommenderAgent":
        return crop_recommender_agent.respond(query)
    elif agent_name == "WeatherForecastAgent":
        return weather_forecast_agent.get_weather_analysis(query)
    elif agent_name == "LocationAgriAssistant":
        return location_agri_assistant.respond(query)
    elif agent_name == "NewsAgent":
        return news_agent.get_agri_news(query)
    elif agent_name == "CreditPolicyMarketAgent":
        return credit_policy_market_agent.respond_to_query(query)
    elif agent_name == "AnswerGraderAgent":
        return answer_grader_agent.grade(query.get("question", ""), query.get("answer", ""))
    elif agent_name == "SynthesizerAgent":
        return synthesizer_agent.synthesize(query.get("responses", []))
    elif agent_name == "CropDiseaseDetectionAgent":
        return crop_disease_agent.analyze_disease(query = query, image_path=image_path)
    elif agent_name == "FactCheckerAgent":
        return fact_checker_agent.score(query)
    elif agent_name == "ImageAnalysisAgent":
        return image_analysis_agent.describe_image(image_path)
    elif agent_name == "MarketPriceAgent":
        return market_price_agent.chat(query)
    elif agent_name == "MultiLanguageTranslatorAgent":
        return multi_language_translator_agent.translate_robust(query)
    elif agent_name == "PestPredictionAgent":
        return pest_prediction_agent.respond(query)
    elif agent_name == "RiskManagementAgent":
        return risk_management_agent.assess(query)
    elif agent_name == "WebScrapingAgent":
        return web_scraping_agent.scrape(query)
    elif agent_name == "CropYieldAgent":
        return crop_yield_assistant.respond(query)
    elif agent_name == "FertilizerRecommenderAgent":
        return fertilizer_recommender_agent.recommend_fertilizer(query)
    else:
        return f"No implementation for agent: {agent_name}"

def call_agent_simple(agent_name: str, query: str, image_path: str = None) -> Dict[str, Any]:
    try:
        agent_response = call_agent(agent_name, query, image_path)
        
        return {
            "agent_name": agent_name,
            "response": agent_response,
        }
    except Exception as e:
        return {
            "agent_name": agent_name,
            "response": f"Error: {str(e)}",
        }

def grade_answer(question: str, answer: str) -> Dict[str, Any]:
    try:
        grade_result = answer_grader_agent.grade(question, answer)
        print(f"Answer Grader Response: {grade_result}")
        return {
            "grade": grade_result.decision,
            "is_good_answer": getattr(grade_result, 'decision', False),
            "reasoning": getattr(grade_result, 'feedback', ''),
            "score": 1 if getattr(grade_result, 'decision', False) else 0
        }
    except Exception as e:
        return {
            "grade": None,
            "is_good_answer": False,
            "reasoning": f"Grading error: {str(e)}",
            "score": 0,
            "error": str(e)
        }

def rag_node(state: MainWorkflowState):
    rag_response = run_adaptive_rag(state["query"])
    documents = []
    extractions = ""
    if isinstance(rag_response, dict):
        documents = rag_response.get("documents", [])
        extractions = rag_response.get("extractions", "")
        generation = rag_response.get("generation", "")
    else:
        generation = rag_response
    
    return {
        "rag_response": rag_response,
        "synthesized_result": rag_response,
        "documents": documents,
        "extractions": extractions,
        "generation": generation,
        "current_mode": "rag"
    }

def router_node(state: MainWorkflowState):
    router_result = run_router_agent(state["query"], state.get("image_path"))
    return {
        "router_result": router_result,
        "current_mode": "tooling"
    }

def agent_calls_node(state: MainWorkflowState):
    agent_responses = {}
    agents = state["router_result"].get("agents", [])
    
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(call_agent_simple, agent, state["query"], state.get("image_path")): agent 
            for agent in agents
        }
        
        for future in as_completed(futures):
            agent_name = futures[future]
            try:
                result = future.result()
                agent_responses[agent_name] = result["response"]
                print(f"Agent {agent_name} Response: {result['response']}")
                
            except Exception as e:
                agent_responses[agent_name] = f"Error: {str(e)}"
                print(f"Agent {agent_name} Error: {str(e)}")
    
    return {
        "agent_responses": agent_responses
    }

def synthesize_tooling_node(state: MainWorkflowState):
    all_responses = []
    
    for agent_name, response in state["agent_responses"].items():
        all_responses.append(response)
    
    synthesized_result = synthesizer_agent.synthesize(all_responses)
    
    return {
        "synthesized_result": synthesized_result
    }

def grading_node(state: MainWorkflowState):
    answer_grade_result = grade_answer(state["query"], str(state["synthesized_result"]))
    
    return {
        "answer_grade": answer_grade_result
    }

def mode_decision_edge(state: MainWorkflowState):
    if state.get("is_image_query", False):
        return "end"
    
    answer_grade = state["answer_grade"]
    current_mode = state["current_mode"]
    has_switched_mode = state.get("has_switched_mode", False)
    
    is_answer_complete = answer_grade.get("is_good_answer", False)
    
    print(f"Quality Assessment - Answer Complete: {is_answer_complete}")
    
    if is_answer_complete:
        return "end"
    
    if current_mode == "tooling":
        return "fallback"
    elif current_mode == "rag":
        return "switch_to_tooling"
    
    return "fallback"

def switch_to_tooling_node(state: MainWorkflowState):
    router_result = run_router_agent(state["query"], state.get("image_path"))
    return {
        "router_result": router_result,
        "current_mode": "tooling",
        "has_switched_mode": True
    }

def fallback_node(state: MainWorkflowState):
    fallback_response = multi_language_translator_agent.respond(f"Provide a general answer for: {state['query']}")
    
    return {
        "synthesized_result": fallback_response
    }

def start_routing_edge(state: MainWorkflowState):
    if state.get("is_image_query", False):
        return "router"
    
    initial_mode = state["initial_mode"]
    if initial_mode == "rag":
        return "rag"
    else:
        return "router"

def build_hybrid_workflow_graph():
    graph = StateGraph(MainWorkflowState)
    
    graph.add_node("rag", rag_node)
    graph.add_node("router", router_node)
    graph.add_node("agent_calls", agent_calls_node)
    graph.add_node("synthesize_tooling", synthesize_tooling_node)
    graph.add_node("grading", grading_node)
    graph.add_node("switch_to_tooling", switch_to_tooling_node)
    graph.add_node("fallback", fallback_node)
    
    graph.add_conditional_edges(START, start_routing_edge, {"rag": "rag", "router": "router"})
    
    graph.add_edge("rag", "grading")
    graph.add_edge("router", "agent_calls")
    graph.add_edge("agent_calls", "synthesize_tooling")
    graph.add_edge("synthesize_tooling", "grading")
    
    graph.add_conditional_edges(
        "grading", 
        mode_decision_edge, 
        {
            "end": END, 
            "switch_to_tooling": "switch_to_tooling",
            "fallback": "fallback"
        }
    )
    
    graph.add_edge("switch_to_tooling", "agent_calls")
    graph.add_edge("fallback", END)
    
    return graph

hybrid_workflow_graph = build_hybrid_workflow_graph()
compiled_hybrid_graph = hybrid_workflow_graph.compile()

def run_workflow(query: str, mode: str = "rag", image_path: str = None) -> Dict[str, Any]:
    if not internet_checker.is_connected() and hf_model:
        hf_response = hf_model.infer(query)
        return {
            "answer": hf_response,
            "answer_quality_grade": {"is_good_answer": True, "reasoning": "Offline mode - HF Model response"},
            "is_answer_complete": True,
            "final_mode": "offline",
            "switched_modes": False,
            "is_image_query": image_path is not None
        }
    
    is_image_query = image_path is not None
    
    state = MainWorkflowState(
        query=query,
        image_path=image_path or "",
        initial_mode=mode,
        current_mode=mode,
        rag_response="",
        router_result={},
        agent_responses={},
        fact_checked_responses={},
        synthesized_result="",
        grading=None,
        answer_grade=None,
        rewritten_query="",
        generation="",
        extractions="",
        documents=[],
        has_switched_mode=False,
        is_image_query=is_image_query
    )
    
    if mode.lower() not in ["rag", "tooling"]:
        raise ValueError("Mode must be either 'rag' or 'tooling'")
    
    final_state = compiled_hybrid_graph.invoke(state)
    
    return {
        "answer": final_state["synthesized_result"],
        "answer_quality_grade": final_state.get("answer_grade", {}),
        "is_answer_complete": final_state.get("answer_grade", {}).get("is_good_answer", False),
        "final_mode": final_state.get("current_mode", mode),
        "switched_modes": final_state.get("has_switched_mode", False),
        "is_image_query": final_state.get("is_image_query", False)
    }

if __name__ == "__main__":
    import time
    questions = [
        "Hello how are you?",
        "Estimate crop yield for wheat in Punjab in winter of 2025",
        "How can farmers manage pest outbreaks in cotton fields?",
        "What is the market price trend for wheat in India?",
        "How to prevent fungal diseases in tomato crops?",
    ]
    
    image_queries = [
        ("Analyze this crop disease", "Images/Crop/crop_disease.jpg"),
        ("Check for pests in this image", "Images/Pests/jpg_0.jpg"),
    ]
    
    mode = input("Select initial mode (rag/tooling): ").strip().lower()
    if mode not in ["rag", "tooling"]:
        print("Invalid mode. Using 'rag' as default.")
        mode = "rag"
    
    query_type = input("Test type (text/image): ").strip().lower()
    
    if query_type == "image":
        test_queries = image_queries
    else:
        test_queries = [(q, None) for q in questions]
    
    for idx, (user_query, image_path) in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Question {idx}: {user_query}")
        if image_path:
            print(f"Image Path: {image_path}")
        print(f"Initial Mode: {mode.upper()}")
        print('='*80)
        
        start_time = time.time()
        result = run_workflow(user_query, mode, image_path)
        end_time = time.time()
        
        print(f"Answer: {result['answer']}")
        print(f"\nQuality Metrics:")
        print(f"  - Is Answer Complete: {result['is_answer_complete']}")
        print(f"  - Final Mode: {result['final_mode']}")
        print(f"  - Switched Modes: {result['switched_modes']}")
        print(f"  - Is Image Query: {result['is_image_query']}")
        print(f"  - Processing Time: {end_time - start_time:.2f}s")
        
        if result['answer_quality_grade'].get('reasoning'):
            print(f"  - Quality Grade Reasoning: {result['answer_quality_grade']['reasoning']}")